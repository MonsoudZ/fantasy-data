"""Dynasty value: redraft value projected forward over a player's career.

Redraft asks "who's best THIS year." Dynasty asks "who's most valuable over the
next several years," so age dominates: a 22-year-old and a 30-year-old with the
same projection are worth very different things because one has a career ahead
and the other is near the cliff.

We estimate age curves from history -- for each position, average points-per-game
by age, normalized to the position's peak -- then value a player as the
discounted sum of his projected production across the next `years` seasons,
scaled by how the curve says he'll age.

    from ffdata.dynasty import dynasty_board
    print(dynasty_board(2024).head(15))   # young studs rise, aging vets fall
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .db import connect
from .draft import POSITIONS, draft_board, _season_agg, _roster_info

_MIN_AGE, _MAX_AGE = 21, 36


def age_curves(con=None, before_season: int | None = None) -> dict:
    """Per-position relative value (0-1) by age.

    Built with the delta method: average the year-over-year PPG change for the
    SAME player aging A -> A+1, then cumulate into a curve. This removes the
    survivorship bias that inflates naive "average PPG by age" (only good players
    survive to older ages, so the raw average never declines).

    `before_season`: if given, only seasons strictly before it feed the curves.
    A dynasty valuation made in the preseason of season S must not learn its age
    curves from S or later (that would be look-ahead); pass ``before_season=S``.
    """
    con = con or connect()
    m = _season_agg(con).merge(_roster_info(con), on=["player_id", "season"], how="inner")
    if before_season is not None:
        # Drop target-or-later seasons before the shift(-1) below, so no A->A+1
        # transition can peek into the season being valued.
        m = m[m["season"] < before_season]
    m = m[m["games"] >= 6].copy()
    m["age"] = m["season"] - m["birth_year"]
    m["ppg"] = m["fp"] / m["games"].clip(lower=1)
    m = m.sort_values(["player_id", "season"])
    m["next_ppg"] = m.groupby("player_id")["ppg"].shift(-1)
    m["next_season"] = m.groupby("player_id")["season"].shift(-1)
    m = m[(m["next_season"] == m["season"] + 1) & (m["age"].between(_MIN_AGE, _MAX_AGE))]
    m["ratio"] = (m["next_ppg"] / m["ppg"].clip(lower=1)).clip(0.4, 1.8)

    ages = np.arange(_MIN_AGE, _MAX_AGE + 1)
    curves = {}
    for pos in POSITIONS:
        g = m[m["position"] == pos].groupby("age")["ratio"]
        step = g.mean().where(g.count() >= 8)                    # A -> A+1 multiplier
        rel = {_MIN_AGE: 1.0}
        for a in ages[:-1]:
            rel[a + 1] = rel[a] * (step.get(a) if pd.notna(step.get(a)) else 1.0)
        peak = max(rel.values())
        curves[pos] = {a: v / peak for a, v in rel.items()}
    return curves


def dynasty_board(target_season: int, years: int = 4, discount: float = 0.85,
                  con=None) -> pd.DataFrame:
    """Rank players by dynasty value = discounted, age-curve-projected redraft value."""
    con = con or connect()
    board = draft_board(target_season, con=con)
    if board.empty:
        return board
    curves = age_curves(con, before_season=target_season)
    ri = _roster_info(con)
    ages = ri[ri["season"] == target_season][["player_id", "birth_year"]]
    board = board.merge(ages, on="player_id", how="left")
    board["age"] = (target_season - board["birth_year"]).fillna(27).astype(int)

    def value(row):
        # Build on VOR (positional scarcity), not raw points, then age it forward.
        curve = curves.get(row["position"], {})
        a0 = int(row["age"])
        base = curve.get(a0) or curve.get(min(max(a0, _MIN_AGE), _MAX_AGE)) or 1.0
        total = 0.0
        for t in range(years):
            rel = curve.get(a0 + t, 0.0) / (base or 1.0)
            total += max(row["vor"], 0.0) * rel * (discount ** t)
        return round(total, 1)

    board["dynasty_value"] = board.apply(value, axis=1)
    cols = ["player", "position", "age", "proj", "vor", "dynasty_value"]
    return board[cols].sort_values("dynasty_value", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    from .ingest import current_nfl_season
    p = argparse.ArgumentParser(prog="python -m ffdata.dynasty", description="Dynasty value board")
    p.add_argument("--season", type=int, default=current_nfl_season())
    p.add_argument("--years", type=int, default=4, help="future seasons to value")
    p.add_argument("--discount", type=float, default=0.85)
    p.add_argument("--n", type=int, default=25)
    args = p.parse_args()
    board = dynasty_board(args.season, years=args.years, discount=args.discount)
    if board.empty:
        raise SystemExit(f"No dynasty data for {args.season}.")
    pd.set_option("display.width", 100)
    print(f"\nDynasty board {args.season} "
          f"(value = {args.years}yr age-curve-projected, discount {args.discount}):\n")
    print(board.head(args.n).to_string(index=False))
