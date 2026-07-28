"""Reproducible experiment metadata, holdout policy, persistence, and reports."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from .data import PACKAGE_ROOT, manifest_path

DEFAULT_POLICY = {
    "version": 1,
    "description": (
        "Regression seasons may be rerun while developing. The locked season is a single-use "
        "final evaluation and must not guide feature selection."
    ),
    "experiments": {
        "weekly": {"regression": 2025, "locked": 2026},
        "draft": {"regression": 2025, "locked": 2026},
        "season": {"regression": 2025, "locked": 2026},
        "season-sweep": {"regression": 2025, "locked": 2026},
        "strategy-sweep": {"regression": 2025, "locked": 2026},
    },
}


class HoldoutError(ValueError):
    """A holdout selection violates the experiment policy."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_experiment_root(
    environ: Mapping[str, str] | None = None,
    *,
    source_root: Path = PACKAGE_ROOT,
) -> Path:
    env = os.environ if environ is None else environ
    if env.get("FFDATA_EXPERIMENTS"):
        return Path(env["FFDATA_EXPERIMENTS"]).expanduser()
    checkout = Path(source_root) / "experiments"
    if checkout.exists():
        return checkout
    return manifest_path().parent / "experiments"


def load_policy(path: Path | None = None) -> dict:
    path = path or (resolve_experiment_root() / "holdouts.json")
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_POLICY))
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy.get("experiments"), dict):
        raise HoldoutError(f"invalid holdout policy: {path}")
    return policy


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        row["_path"] = path
        rows.append(row)
    return rows


def resolve_holdout(
    kind: str,
    requested: str,
    *,
    policy: dict,
    results_dir: Path,
    consume_locked: bool = False,
    force_locked: bool = False,
) -> tuple[int, str]:
    try:
        spec = policy["experiments"][kind]
    except KeyError as exc:
        raise HoldoutError(f"no holdout policy for experiment {kind!r}") from exc
    if requested in ("regression", "locked"):
        season, tier = int(spec[requested]), requested
    else:
        try:
            season = int(requested)
        except ValueError as exc:
            raise HoldoutError("holdout must be 'regression', 'locked', or a season") from exc
        tier = next((name for name in ("regression", "locked")
                     if int(spec[name]) == season), "exploratory")

    if tier == "locked":
        if not consume_locked:
            raise HoldoutError(
                f"{season} is the locked {kind} holdout; pass --consume-locked-holdout "
                "only for the final evaluation"
            )
        consumed = [r for r in load_results(results_dir)
                    if r.get("kind") == kind and r.get("holdout", {}).get("tier") == "locked"]
        if consumed and not force_locked:
            raise HoldoutError(
                f"the locked {kind} holdout was already consumed by {consumed[-1]['_path'].name}; "
                "use --force-locked-holdout only to reproduce that exact run"
            )
    return season, tier


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap interval for a mean, with deterministic resampling."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(resamples, arr.size))
    estimates = arr[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return {
        "estimate": round(float(arr.mean()), 4),
        "low": round(float(low), 4),
        "high": round(float(high), 4),
        "confidence": confidence,
        "resamples": resamples,
        "n": int(arr.size),
    }


def bootstrap_cluster_mean_ci(
    clusters: Iterable[Iterable[float]],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Percentile interval for a mean, resampling whole clusters.

    Each cluster contributes equally through its within-cluster mean. For season
    simulations a cluster is one NFL season and its observations are draft slots,
    so the interval reflects variation across seasons instead of pretending that
    twelve replays of one real season are twelve independent football histories.
    """
    arrays = [np.asarray(list(cluster), dtype=float) for cluster in clusters]
    if not arrays or any(arr.size == 0 for arr in arrays):
        raise ValueError("cannot bootstrap empty clusters")
    means = np.asarray([arr.mean() for arr in arrays])
    result = bootstrap_mean_ci(
        means, confidence=confidence, resamples=resamples, seed=seed,
    )
    result.update(n_clusters=len(arrays), n=int(sum(arr.size for arr in arrays)))
    return result


def _run_git(args: list[str], root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def git_metadata(root: Path = PACKAGE_ROOT) -> dict:
    commit = _run_git(["rev-parse", "HEAD"], root)
    branch = _run_git(["branch", "--show-current"], root)
    status = _run_git(["status", "--porcelain"], root)
    paths = [line[3:] for line in status.splitlines()] if status else []
    return {"commit": commit, "branch": branch, "dirty": bool(paths), "dirty_paths": paths}


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def data_metadata(path: Path | None = None) -> dict:
    path = path or manifest_path()
    if not path.exists():
        return {"manifest_path": str(path), "available": False}
    blob = path.read_bytes()
    manifest = json.loads(blob)
    assets = {name: entry.get("sha256") for name, entry in sorted(manifest.get("files", {}).items())}
    asset_blob = json.dumps(assets, sort_keys=True, separators=(",", ":")).encode()
    return {
        "manifest_path": str(path),
        "available": True,
        "manifest_version": manifest.get("manifest_version"),
        "manifest_sha256": _sha256_bytes(blob),
        "asset_set_sha256": _sha256_bytes(asset_blob),
        "asset_count": len(assets),
        "assets": assets,
    }


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_record(
    kind: str,
    season: int,
    tier: str,
    config: dict,
    metrics: dict,
    *,
    duration_seconds: float,
    git: dict | None = None,
    data: dict | None = None,
) -> dict:
    return jsonable({
        "schema_version": 1,
        "experiment_id": uuid.uuid4().hex,
        "created_at": utc_now(),
        "kind": kind,
        "holdout": {"season": season, "tier": tier},
        "config": config,
        "metrics": metrics,
        "duration_seconds": round(float(duration_seconds), 3),
        "git": git if git is not None else git_metadata(),
        "data": data if data is not None else data_metadata(),
    })


def _atomic_write_text(text: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def save_record(record: dict, results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    stamp = record["created_at"].replace("-", "").replace(":", "").split(".")[0].replace("+0000", "")
    name = (f"{stamp}Z-{record['kind']}-{record['holdout']['season']}-"
            f"{record['experiment_id'][:8]}.json")
    path = results_dir / name
    _atomic_write_text(json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n", path)
    return path


def _fmt_ci(ci: dict, *, percent: bool = False) -> str:
    scale, suffix = (100, "%") if percent else (1, "")
    return (f"{ci['estimate'] * scale:.1f}{suffix} "
            f"[{ci['low'] * scale:.1f}, {ci['high'] * scale:.1f}]")


def metric_summary(record: dict) -> str:
    metrics, kind = record.get("metrics", {}), record.get("kind")
    if kind == "weekly":
        models = metrics.get("models", {})
        preferred = models.get("lightgbm") or next(iter(models.values()), {})
        return (f"LightGBM MAE {preferred.get('MAE', '—')}, RMSE {preferred.get('RMSE', '—')}, "
                f"rank {preferred.get('weekly_spearman', '—')}")
    if kind == "draft":
        return (f"rank {metrics.get('model_spearman', '—')} vs prior "
                f"{metrics.get('lastyear_spearman', '—')}; MAE {metrics.get('model_mae', '—')}")
    if kind == "season":
        cis = metrics.get("confidence_intervals", {})
        if cis:
            return (f"finish {_fmt_ci(cis['mean_finish'])}; playoffs "
                    f"{_fmt_ci(cis['playoff_rate'], percent=True)}; titles "
                    f"{_fmt_ci(cis['title_rate'], percent=True)}")
        return (f"finish {metrics.get('mean_finish', '—')}; playoffs "
                f"{metrics.get('playoff_rate', '—')}; titles {metrics.get('title_rate', '—')}")
    if kind == "season-sweep":
        fields = metrics.get("field_strengths", [])
        if fields:
            first, last = fields[0], fields[-1]
            def compact(row):
                return (f"finish {row.get('mean_finish', '—')}, "
                        f"playoffs {row.get('playoff_rate', 0) * 100:.1f}%, "
                        f"titles {row.get('title_rate', 0) * 100:.1f}%")

            return (f"{first.get('sharp_fraction', 0) * 100:.0f}% sharp: {compact(first)}; "
                    f"{last.get('sharp_fraction', 1) * 100:.0f}%: {compact(last)}")
    if kind == "strategy-sweep":
        results = metrics.get("strategy_results", {})
        baseline = results.get("baseline", {}).get("field_strengths", [])
        adaptive = results.get("adaptive", {}).get("field_strengths", [])
        paired = metrics.get("paired_adaptive_vs_baseline", [])
        if baseline and adaptive and paired:
            base, challenger, delta = baseline[-1], adaptive[-1], paired[-1]
            finish_delta = delta.get("mean_finish_delta", {}).get("estimate", 0)
            return (f"{challenger.get('sharp_fraction', 1) * 100:.0f}% sharp adaptive: "
                    f"finish {challenger.get('mean_finish', '—')} vs "
                    f"{base.get('mean_finish', '—')} (Δ {finish_delta:+.2f}); playoffs "
                    f"{challenger.get('playoff_rate', 0) * 100:.1f}%; titles "
                    f"{challenger.get('title_rate', 0) * 100:.1f}%")
    return json.dumps(metrics, sort_keys=True)[:160]


def variant_summary(record: dict) -> str:
    """Compact configuration label so unlike experiment variants stay distinct."""
    config, kind = record.get("config", {}), record.get("kind")
    if kind == "season":
        strategy = config.get("draft_strategy", "baseline")
        return f"{config.get('opponent', '—')} · {strategy}"
    if kind == "draft":
        return f"{config.get('board', '—')} · {config.get('scoring', '—')}"
    if kind == "weekly":
        return "LightGBM vs trailing"
    if kind == "season-sweep":
        seasons = config.get("seasons", [])
        fractions = config.get("sharp_fractions", [])
        season_label = (f"{min(seasons)}–{max(seasons)}" if seasons else "—")
        fraction_label = (f"{min(fractions) * 100:.0f}–{max(fractions) * 100:.0f}% sharp"
                          if fractions else "—")
        return f"{season_label} · {fraction_label} · {config.get('draft_strategy', 'baseline')}"
    if kind == "strategy-sweep":
        seasons = config.get("seasons", [])
        fractions = config.get("sharp_fractions", [])
        season_label = (f"{min(seasons)}–{max(seasons)}" if seasons else "—")
        fraction_label = (f"{min(fractions) * 100:.0f}–{max(fractions) * 100:.0f}% sharp"
                          if fractions else "—")
        return f"{season_label} · {fraction_label} · paired"
    return "—"


def render_summary(results_dir: Path, output: Path | None = None) -> str:
    results_dir = Path(results_dir)
    records = sorted(load_results(results_dir), key=lambda r: r.get("created_at", ""), reverse=True)
    lines = [
        "# Experiment results",
        "",
        "Generated from the JSON artifacts in `results/`. Regression holdouts may be rerun; "
        "locked holdouts are reserved for one final evaluation.",
        "",
    ]
    if not records:
        lines.append("No experiment results recorded yet.")
    else:
        lines.extend([
            "| Date | Experiment | Variant | Holdout | Code | Result |",
            "|---|---|---|---|---|---|",
        ])
        for row in records:
            git = row.get("git", {})
            commit = (git.get("commit") or "unknown")[:8] + (" dirty" if git.get("dirty") else "")
            holdout = row.get("holdout", {})
            date = row.get("created_at", "")[:10]
            lines.append(
                f"| {date} | {row.get('kind')} | {variant_summary(row)} | "
                f"{holdout.get('season')} "
                f"({holdout.get('tier')}) | `{commit}` | {metric_summary(row)} |"
            )
    text = "\n".join(lines) + "\n"
    if output is not None:
        _atomic_write_text(text, Path(output))
    return text


def execute(
    runner: Callable[[], dict],
    *,
    kind: str,
    season: int,
    tier: str,
    config: dict,
    clock: Callable[[], float],
    git_reader: Callable[[], dict] = git_metadata,
    data_reader: Callable[[], dict] = data_metadata,
) -> dict:
    before_git, before_data = git_reader(), data_reader()
    if not before_data.get("available"):
        raise HoldoutError(
            "no provenance manifest is available; run the data ingest before recording experiments"
        )
    start = clock()
    metrics = runner()
    duration = clock() - start
    after_git, after_data = git_reader(), data_reader()
    if before_git != after_git:
        raise RuntimeError("Git state changed while the experiment was running; result not recorded")
    if before_data.get("asset_set_sha256") != after_data.get("asset_set_sha256"):
        raise RuntimeError("data manifest changed while the experiment was running; result not recorded")
    return build_record(
        kind, season, tier, config, metrics, duration_seconds=duration,
        git=before_git, data=before_data,
    )
