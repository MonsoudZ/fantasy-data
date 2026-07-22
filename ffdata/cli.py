"""CLI: python -m ffdata.cli --seasons 2020-2024 --datasets weekly schedules

Defaults pull everything except play-by-play (large; opt in with --pbp).
"""

from __future__ import annotations

import argparse
import sys

from .ingest import current_nfl_season, ingest
from .sources import SOURCES

# Weekly stats begin at 2019 under nflverse's `stats_player_week` asset; earlier
# seasons 404 on that path, so that is the default floor (matches the docs).
_SEASON_FLOOR = 2019


def parse_seasons(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        a, b = int(a), int(b)
        if a > b:
            raise ValueError(f"season range start {a} is after end {b}")
        return list(range(a, b + 1))
    return [int(s) for s in spec.split(",")]


def main() -> None:
    default_datasets = [d for d in SOURCES if d != "pbp"]
    p = argparse.ArgumentParser(description="Ingest nflverse data")
    p.add_argument("--seasons", default=f"{_SEASON_FLOOR}-{current_nfl_season()}")
    p.add_argument("--datasets", nargs="*", default=default_datasets, choices=list(SOURCES))
    p.add_argument("--pbp", action="store_true", help="include play-by-play")
    p.add_argument("--force", action="store_true", help="re-download cached files")
    args = p.parse_args()

    datasets = list(args.datasets)
    if args.pbp and "pbp" not in datasets:
        datasets.append("pbp")

    seasons = parse_seasons(args.seasons)
    print(f"Ingesting {datasets} for seasons {seasons[0]}-{seasons[-1]}")
    failures = ingest(datasets, seasons, force=args.force)
    if failures:
        print(f"\n{len(failures)} download(s) failed:", file=sys.stderr)
        for target, err in failures:
            print(f"  - {target}: {err}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
