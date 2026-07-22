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
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .db import RAW
from .sources import SOURCES

# Weekly stats begin at 2019 under nflverse's `stats_player_week` asset; earlier
# seasons 404 on that path. The shared data floor for CLIs and feature builds.
FIRST_SEASON = 2019
UA = {"User-Agent": "ff-data-ingest/0.1"}

# Offensive positions kept from the (all-position) weekly file.
# Offensive skill positions, plus K -- kdst.build_kicker reads kickers out of
# `weekly`, so dropping them here would silently kill kicker projections.
# Team defense comes from `schedules`, not here.
_KEEP_POSITIONS = {"QB", "RB", "WR", "TE", "FB", "K"}


def _normalize_weekly(df: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """Map nflverse's `stats_player_week` file onto the schema our code expects.

    The new asset renamed a couple of columns and now includes every position.
    Rename them back and keep the positions we actually model (skill + kickers).
    """
    df = df.rename(columns={"team": "recent_team", "passing_interceptions": "interceptions"})
    if "position" in df.columns:
        df = df[df["position"].isin(_KEEP_POSITIONS)].reset_index(drop=True)
    return df


def _normalize_depth_charts(df: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """Depth charts changed format mid-stream: older files are season/week rows,
    newer ones are dated LIVE snapshots with no season column and many snapshots
    stacked together. Stamp the season we asked for, and for snapshot files keep
    only the most recent chart (the current depth order)."""
    df = df.copy()
    if season is not None and ("season" not in df.columns or df["season"].isna().all()):
        df["season"] = season
    if "dt" in df.columns and df["dt"].notna().any():
        df = df[df["dt"] == df["dt"].max()].reset_index(drop=True)
    return df


# Per-dataset transforms applied after download, before writing parquet.
NORMALIZERS = {"weekly": _normalize_weekly, "depth_charts": _normalize_depth_charts}


def current_nfl_season(today: dt.date | None = None) -> int:
    """NFL seasons are labeled by their starting year; new data begins ~Sept."""
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


def upcoming_nfl_season(today: dt.date | None = None) -> int:
    """The season you'd DRAFT for: the one in progress once games start, else the
    one about to kick off.

    `current_nfl_season` names the most recent season with games, which in the
    offseason is the one already finished -- drafting against it would rank
    players for a season that's already over.
    """
    today = today or dt.date.today()
    cur = current_nfl_season(today)
    return cur if today.month >= 9 else cur + 1


def season_not_started(season: int, today: dt.date | None = None) -> bool:
    """True when `season` has no played games yet.

    `weekly`, `injuries` and `snap_counts` only exist for seasons that have been
    PLAYED, so every weekly tool is dark until Week 1 is in the books. Callers use
    this to say so plainly instead of failing on an empty frame -- or, worse,
    quietly serving last season's numbers under this season's label.
    """
    return season > current_nfl_season(today)


NOT_STARTED_HINT = (
    "Weekly projections need games that have been played. Draft, keepers, "
    "trades, dynasty and game lines all work in the offseason."
)


def _download(url: str, retries: int = 3, backoff: float = 2.0) -> bytes:
    """Fetch bytes, retrying transient failures with exponential backoff.

    Connection errors, timeouts, and 5xx responses are retried; 4xx (missing
    asset, policy denial) fail fast because a retry won't change the outcome.
    """
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(backoff * (2 ** attempt))
    raise RuntimeError("unreachable")  # pragma: no cover


def _fetch_to_parquet(url: str, dest: Path, normalize=None, season: int | None = None) -> int:
    """Fetch a remote csv/parquet and store it as parquet. Returns row count."""
    blob = _download(url)
    if url.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(blob), low_memory=False)
    else:
        df = pd.read_parquet(io.BytesIO(blob))
    if normalize is not None:
        df = normalize(df, season)
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
                rows = _fetch_to_parquet(url, dest, normalize=NORMALIZERS.get(name), season=season)
                log(f"  ok    {name}{suffix}: {rows:,} rows")
            except Exception as exc:  # noqa: BLE001 - collect and report at the end
                # `log` may be any callable; don't assume it accepts a `file=`
                # kwarg. Route the human-readable line through it, and return the
                # failure so the caller can decide the exit status.
                log(f"  FAIL  {name}{suffix}: {exc}")
                failures.append((f"{name}{suffix}", str(exc)))
    return failures
