"""CLI: python -m ffdata.cli --seasons 2020-2024 --datasets weekly schedules

Defaults pull everything except play-by-play (large; opt in with --pbp).
"""

from __future__ import annotations

import argparse
import sys

from .ingest import FIRST_SEASON, current_nfl_season, ingest
from .sources import SOURCES


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
    p.add_argument("--seasons", default=f"{FIRST_SEASON}-{current_nfl_season()}")
    p.add_argument("--datasets", nargs="*", default=default_datasets, choices=list(SOURCES))
    p.add_argument("--pbp", action="store_true", help="include play-by-play")
    p.add_argument("--force", action="store_true", help="re-download cached files")
    p.add_argument("--live", action="store_true",
                   help="also refresh Sleeper's live availability feed (today's "
                        "IR/PUP/suspensions); nflverse only knows last season")
    args = p.parse_args()

    datasets = list(args.datasets)
    if args.pbp and "pbp" not in datasets:
        datasets.append("pbp")

    seasons = parse_seasons(args.seasons)
    print(f"Ingesting {datasets} for seasons {seasons[0]}-{seasons[-1]}")
    failures = ingest(datasets, seasons, force=args.force)

    if args.live:
        from .sleeper import refresh_live_status
        print("Refreshing live availability (Sleeper)")
        try:
            refresh_live_status(force=args.force)
        except Exception as exc:  # noqa: BLE001 - a live extra must never fail ingest
            print(f"  FAIL  sleeper_status: {exc}", file=sys.stderr)
            failures.append(("sleeper_status", str(exc)))

    if failures:
        print(f"\n{len(failures)} download(s) failed:", file=sys.stderr)
        for target, err in failures:
            print(f"  - {target}: {err}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
