#!/usr/bin/env python
"""Ingest a consensus-ADP rankings PDF into the lake as the `consensus_adp` view.

Fantasy sites publish a one-page "team depth chart" PDF where every player is
printed as `SLOT Player Name (overall rank)`. That overall rank IS the consensus
ADP -- the market's view of who goes where, which is a cleaner "who's the starter"
signal than either last year's points or our own projection. We parse the
name/rank pairs (the team-column layout is unreliable from text extraction, but
`Name (rank)` is not) and write one row per player.

    pip install ff-data[adp]
    python scripts/ingest_adp.py ~/Downloads/NFL26_CS_Depth.pdf --source CBS

Rerun whenever the source updates; it overwrites the view's single parquet.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# `SLOT Player Name (rank)` -- SLOT is QB1/RB2/WR1/TE1/K etc. Names carry ., ', -
# and suffixes (Jr., III). Rank is the parenthesised overall/ADP number.
_ENTRY = re.compile(r"(?:QB|RB|WR|TE|K|DST|D/ST)\d?\s+"
                    r"([A-Z][A-Za-z.'\-]+(?:\s+[A-Za-z.'\-]+)*?)\s+\((\d+)\)")


def parse_pdf(pdf_path: Path) -> pd.DataFrame:
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("pypdf is required: pip install ff-data[adp]")
    text = "\n".join((p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages)
    pairs = _ENTRY.findall(text)
    if not pairs:
        sys.exit(f"No `Name (rank)` entries found in {pdf_path} -- wrong PDF layout?")
    # Keep each player's BEST (lowest) rank if a name somehow repeats.
    best: dict[str, int] = {}
    for name, rank in pairs:
        r = int(rank)
        if name not in best or r < best[name]:
            best[name] = r
    df = pd.DataFrame(sorted(best.items(), key=lambda kv: kv[1]), columns=["player", "adp"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a consensus-ADP rankings PDF.")
    ap.add_argument("pdf", type=Path, help="path to the rankings/depth-chart PDF")
    ap.add_argument("--source", default="consensus", help="label stored with the rows")
    args = ap.parse_args()
    if not args.pdf.exists():
        sys.exit(f"no such file: {args.pdf}")

    df = parse_pdf(args.pdf)
    df["source"] = args.source

    from ffdata.db import RAW
    dest = RAW / "consensus_adp" / "consensus_adp.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"wrote {len(df)} players -> {dest}  (ranks {df['adp'].min()}-{df['adp'].max()})")


if __name__ == "__main__":
    main()
