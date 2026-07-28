"""Data-lake paths, validation, atomic publishing, and provenance.

The source checkout historically stored parquet under ``data/raw``.  Keep using
that lake when it already exists, but do not assume an installed wheel can write
beside its package files: fresh installs default to ``~/.ff-data/raw``.

Set ``FFDATA_DATA`` to an exact raw-lake path, or ``FFDATA_HOME`` to a base
directory whose ``raw`` child will hold the lake.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow.parquet as pq

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
LEGACY_RAW = PACKAGE_ROOT / "data" / "raw"
MANIFEST_VERSION = 1
SCHEMA_CONTRACT_VERSION = 1


class DataValidationError(ValueError):
    """Downloaded data does not satisfy the source's minimum contract."""


def resolve_raw_path(
    environ: Mapping[str, str] | None = None,
    *,
    legacy_raw: Path | None = None,
    user_home: Path | None = None,
) -> Path:
    """Resolve the raw lake with explicit configuration taking precedence.

    ``FFDATA_DATA`` names the raw directory itself. ``FFDATA_HOME`` names its
    parent. The legacy checkout lake wins only when it already exists; this lets
    existing editable installs continue to work while normal installations use
    a user-writable location.
    """
    env = os.environ if environ is None else environ
    if env.get("FFDATA_DATA"):
        return Path(env["FFDATA_DATA"]).expanduser()
    if env.get("FFDATA_HOME"):
        return Path(env["FFDATA_HOME"]).expanduser() / "raw"
    legacy = LEGACY_RAW if legacy_raw is None else Path(legacy_raw)
    if legacy.exists():
        return legacy
    home = Path.home() if user_home is None else Path(user_home)
    return home / ".ff-data" / "raw"


RAW = resolve_raw_path()


def manifest_path(raw: Path = RAW) -> Path:
    """The manifest sits beside ``raw`` so DuckDB never treats it as a dataset."""
    return Path(raw).parent / "manifest.json"


def validate_frame(
    dataset: str,
    frame: pd.DataFrame,
    contract: Mapping | None,
    *,
    season: int | None = None,
) -> None:
    """Validate a frame against a small, stable per-source schema contract."""
    if frame.empty:
        raise DataValidationError(f"{dataset}: downloaded frame is empty")
    contract = contract or {}
    columns = set(frame.columns)
    missing = set(contract.get("required", ())) - columns
    if missing:
        raise DataValidationError(f"{dataset}: missing required columns: {sorted(missing)}")
    for alternatives in contract.get("any_of", ()):
        if not columns.intersection(alternatives):
            raise DataValidationError(
                f"{dataset}: expected one of these columns: {sorted(alternatives)}"
            )
    if season is not None:
        if "season" not in columns:
            raise DataValidationError(f"{dataset}: seasonal asset has no season column")
        actual = set(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int).unique())
        if not actual:
            raise DataValidationError(f"{dataset}: seasonal asset has no valid season values")
        if actual and actual != {int(season)}:
            raise DataValidationError(
                f"{dataset}: requested season {season}, downloaded seasons {sorted(actual)}"
            )


def _temporary_path(dest: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    os.close(fd)
    return Path(name)


def atomic_write_parquet(frame: pd.DataFrame, dest: Path) -> None:
    """Write and verify parquet before atomically replacing ``dest``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary_path(dest)
    try:
        frame.to_parquet(tmp, index=False)
        metadata = pq.read_metadata(tmp)
        if metadata.num_rows != len(frame):
            raise RuntimeError(
                f"parquet verification failed: wrote {metadata.num_rows} rows, expected {len(frame)}"
            )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return version("ff-data")
    except PackageNotFoundError:  # pragma: no cover - only an uninstalled source tree
        return "unknown"


def _load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"cannot read provenance manifest {path}: {exc}") from exc


def _atomic_write_json(payload: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary_path(dest)
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _record_provenance(
    dataset: str,
    dest: Path,
    *,
    rows: int,
    columns: list[str],
    source: str,
    season: int | None,
    raw: Path,
    retrieved_at: str | None,
    provenance: str,
) -> None:
    raw, dest = Path(raw), Path(dest)
    path = manifest_path(raw)
    manifest = _load_manifest(path)
    now = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("manifest_version", MANIFEST_VERSION)
    manifest["updated_at"] = now
    manifest["ffdata_version"] = _package_version()
    files = manifest.setdefault("files", {})
    try:
        key = str(dest.relative_to(raw))
    except ValueError:
        key = str(dest)
    stat = dest.stat()
    files[key] = {
        "dataset": dataset,
        "season": season,
        "source": source,
        "retrieved_at": retrieved_at or now,
        "recorded_at": now,
        "provenance": provenance,
        "schema_contract_version": SCHEMA_CONTRACT_VERSION,
        "rows": rows,
        "columns": columns,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(dest),
    }
    _atomic_write_json(manifest, path)


def record_provenance(
    dataset: str,
    dest: Path,
    frame: pd.DataFrame,
    *,
    source: str,
    season: int | None = None,
    raw: Path = RAW,
    retrieved_at: str | None = None,
    provenance: str = "download",
) -> None:
    """Atomically add or replace one file's entry in the lake manifest."""
    _record_provenance(
        dataset, dest, rows=len(frame), columns=list(frame.columns), source=source,
        season=season, raw=raw, retrieved_at=retrieved_at, provenance=provenance,
    )


def register_cached_parquet(
    dataset: str,
    dest: Path,
    *,
    source: str,
    contract: Mapping | None = None,
    season: int | None = None,
    raw: Path = RAW,
) -> None:
    """Validate and backfill provenance for a previously cached parquet.

    A matching size/mtime/contract-version entry is already verified and is a
    no-op. Older lakes pay the validation/hash cost once, not on every ingest.
    """
    raw, dest = Path(raw), Path(dest)
    stat = dest.stat()
    path = manifest_path(raw)
    try:
        key = str(dest.relative_to(raw))
    except ValueError:
        key = str(dest)
    existing = _load_manifest(path).get("files", {}).get(key, {})
    if (
        existing.get("bytes") == stat.st_size
        and existing.get("mtime_ns") == stat.st_mtime_ns
        and existing.get("schema_contract_version") == SCHEMA_CONTRACT_VERSION
    ):
        return

    metadata = pq.read_metadata(dest)
    columns = metadata.schema.to_arrow_schema().names
    contract = contract or {}
    missing = set(contract.get("required", ())) - set(columns)
    if metadata.num_rows == 0:
        raise DataValidationError(f"{dataset}: cached parquet is empty")
    if missing:
        raise DataValidationError(f"{dataset}: missing required columns: {sorted(missing)}")
    for alternatives in contract.get("any_of", ()):
        if not set(columns).intersection(alternatives):
            raise DataValidationError(
                f"{dataset}: expected one of these columns: {sorted(alternatives)}"
            )
    if season is not None:
        if "season" not in columns:
            raise DataValidationError(f"{dataset}: seasonal asset has no season column")
        season_values = pq.read_table(dest, columns=["season"])["season"].to_pylist()
        actual = {int(v) for v in season_values if v is not None}
        if not actual:
            raise DataValidationError(f"{dataset}: seasonal asset has no valid season values")
        if actual != {int(season)}:
            raise DataValidationError(
                f"{dataset}: requested season {season}, cached seasons {sorted(actual)}"
            )

    retrieved = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    _record_provenance(
        dataset, dest, rows=metadata.num_rows, columns=columns, source=source,
        season=season, raw=raw, retrieved_at=retrieved,
        provenance="backfilled_from_file_mtime",
    )


def publish_parquet(
    dataset: str,
    frame: pd.DataFrame,
    dest: Path,
    *,
    source: str,
    contract: Mapping | None = None,
    season: int | None = None,
    raw: Path = RAW,
) -> None:
    """Validate, atomically publish, and record provenance for one dataset."""
    validate_frame(dataset, frame, contract, season=season)
    atomic_write_parquet(frame, dest)
    record_provenance(dataset, dest, frame, source=source, season=season, raw=raw)
