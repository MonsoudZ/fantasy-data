"""Platform-agnostic fantasy scoring.

Because you play across multiple platforms, raw stats are the source of truth
and scoring is just a config. Define one ScoringRules per league; the same
weekly dataset scores them all. nflverse ships precomputed `fantasy_points`
and `fantasy_points_ppr` columns too -- `score()` lets you match any custom
league (TE premium, 6-pt passing TDs, etc.) and is the column your models
should train against.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScoringRules:
    pass_yd: float = 0.04       # 1 pt / 25 yds
    pass_td: float = 4.0
    interception: float = -2.0
    rush_yd: float = 0.1        # 1 pt / 10 yds
    rush_td: float = 6.0
    reception: float = 1.0      # PPR by default
    rec_yd: float = 0.1
    rec_td: float = 6.0
    te_reception_bonus: float = 0.0   # TE premium leagues
    fumble_lost: float = -2.0
    two_pt: float = 2.0
    special_teams_td: float = 6.0


PPR = ScoringRules()
HALF_PPR = ScoringRules(reception=0.5)
STANDARD = ScoringRules(reception=0.0)


def score(weekly: pd.DataFrame, rules: ScoringRules, col: str = "fp") -> pd.DataFrame:
    """Append a fantasy-point column computed from raw weekly stats."""
    g = lambda c: weekly.get(c, pd.Series(0, index=weekly.index)).fillna(0)

    pts = (
        g("passing_yards") * rules.pass_yd
        + g("passing_tds") * rules.pass_td
        + g("interceptions") * rules.interception
        + g("rushing_yards") * rules.rush_yd
        + g("rushing_tds") * rules.rush_td
        + g("receptions") * rules.reception
        + g("receiving_yards") * rules.rec_yd
        + g("receiving_tds") * rules.rec_td
        + (g("rushing_fumbles_lost") + g("receiving_fumbles_lost") + g("sack_fumbles_lost"))
        * rules.fumble_lost
        + (g("passing_2pt_conversions") + g("rushing_2pt_conversions") + g("receiving_2pt_conversions"))
        * rules.two_pt
        + g("special_teams_tds") * rules.special_teams_td
    )
    if rules.te_reception_bonus and "position" in weekly:
        pts = pts + g("receptions") * rules.te_reception_bonus * (weekly["position"] == "TE")

    out = weekly.copy()
    out[col] = pts.round(2)
    return out
