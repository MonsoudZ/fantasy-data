"""Download nflverse datasets into a local data lake.

Layout:
    data/raw/<dataset>/<dataset>_<season>.parquet   (seasonal datasets)
    data/raw/<dataset>/<dataset>.parquet            (non-seasonal datasets)

Idempotent: existing files are skipped unless force=True, except the current
season, which is always refreshed (stats update nightly during the season).
"""

from __future__ import annotations

import datetime as dt
import io
import urllib.request
from pathlib import Path

import pandas as pd

from .sources import SOURCES

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
UA = {"User-Agent": "ff-data-ingest/0.1"}

# Offensive positions kept from the (all-position) weekly file.
_OFFENSE = {"QB", "RB", "WR", "TE", "FB"}


def _normalize_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Map nflverse's `stats_player_week` file onto the schema our code expects.

    The new asset renamed a couple of columns and now includes every position
    (defense, kickers). Rename them back and keep offensive players only.
    """
    df = df.rename(columns={"team": "recent_team", "passing_interceptions": "interceptions"})
    if "position" in df.columns:
        df = df[df["position"].isin(_OFFENSE)].reset_index(drop=True)
    return df


# Per-dataset transforms applied after download, before writing parquet.
NORMALIZERS = {"weekly": _normalize_weekly}


def current_nfl_season(today: dt.date | None = None) -> int:
    """NFL seasons are labeled by their starting year; new data begins ~Sept."""
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _fetch_to_parquet(url: str, dest: Path, normalize=None) -> int:
    """Fetch a remote csv/parquet and store it as parquet. Returns row count."""
    blob = _download(url)
    if url.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(blob), low_memory=False)
    else:
        df = pd.read_parquet(io.BytesIO(blob))
    if normalize is not None:
        df = normalize(df)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return len(df)


def ingest(
    datasets: list[str],
    seasons: list[int],
    force: bool = False,
    log=print,
) -> list[tuple[str, str]]:
    """Download the requested datasets into the lake.

    Returns a list of ``(target, error)`` for every download that failed. The
    list is empty on a fully successful run; callers (see ``cli.main``) treat a
    non-empty list as a failure and exit non-zero, so a fully-failed ingest is
    never mistaken for success.
    """
    this_season = current_nfl_season()
    failures: list[tuple[str, str]] = []
    for name in datasets:
        spec = SOURCES[name]
        if spec["seasonal"]:
            targets = [(s, spec["url"].format(season=s)) for s in seasons]
        else:
            targets = [(None, spec["url"])]

        for season, url in targets:
            suffix = f"_{season}" if season else ""
            dest = RAW / name / f"{name}{suffix}.parquet"
            # Non-seasonal sources (season is None) are single living files that
            # update continuously in-season (schedules: results + Vegas lines),
            # so they must always refresh -- never treat them as cached.
            refresh = force or season is None or season == this_season
            if dest.exists() and not refresh:
                log(f"  skip  {dest.relative_to(RAW.parent.parent)} (cached)")
                continue
            try:
                rows = _fetch_to_parquet(url, dest, normalize=NORMALIZERS.get(name))
                log(f"  ok    {name}{suffix}: {rows:,} rows")
            except Exception as exc:  # noqa: BLE001 - collect and report at the end
                # `log` may be any callable; don't assume it accepts a `file=`
                # kwarg. Route the human-readable line through it, and return the
                # failure so the caller can decide the exit status.
                log(f"  FAIL  {name}{suffix}: {exc}")
                failures.append((f"{name}{suffix}", str(exc)))
    return failures
