"""CLI: python -m ffdata.cli --seasons 2020-2024 --datasets weekly schedules

Defaults pull everything except play-by-play (large; opt in with --pbp).
"""

from __future__ import annotations

import argparse

from .ingest import current_nfl_season, ingest
from .sources import SOURCES


def parse_seasons(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def main() -> None:
    default_datasets = [d for d in SOURCES if d != "pbp"]
    p = argparse.ArgumentParser(description="Ingest nflverse data")
    p.add_argument("--seasons", default=f"2018-{current_nfl_season()}")
    p.add_argument("--datasets", nargs="*", default=default_datasets, choices=list(SOURCES))
    p.add_argument("--pbp", action="store_true", help="include play-by-play")
    p.add_argument("--force", action="store_true", help="re-download cached files")
    args = p.parse_args()

    datasets = list(args.datasets)
    if args.pbp and "pbp" not in datasets:
        datasets.append("pbp")

    seasons = parse_seasons(args.seasons)
    print(f"Ingesting {datasets} for seasons {seasons[0]}-{seasons[-1]}")
    ingest(datasets, seasons, force=args.force)


if __name__ == "__main__":
    main()
