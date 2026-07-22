"""Platform-agnostic fantasy scoring.

Because you play across multiple platforms, raw stats are the source of truth
and scoring is just a config. Define one ScoringRules per league; the same
weekly dataset scores them all. nflverse ships precomputed `fantasy_points`
and `fantasy_points_ppr` columns too -- `score()` lets you match any custom
league (TE premium, 6-pt passing TDs, etc.) and is the column your models
should train against.
"""

from __future__ import annotations

import dataclasses
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
    # --- Kicker (defaults are the near-universal distance ladder) ---
    fg_0_39: float = 3.0        # made field goal, 0-39 yds
    fg_40_49: float = 4.0       # made field goal, 40-49 yds
    fg_50_plus: float = 5.0     # made field goal, 50+ yds
    pat: float = 1.0            # made extra point
    fg_miss: float = 0.0        # missed field goal (some leagues -1)
    # --- Team defense / special teams (DST) ---
    dst_sack: float = 1.0
    dst_int: float = 2.0
    dst_fumble_rec: float = 2.0
    dst_td: float = 6.0         # defensive or return TD
    dst_safety: float = 2.0
    dst_block: float = 2.0      # blocked kick/punt
    # Points-allowed uses a fixed standard tier ladder (see kdst._dst_pa_points),
    # not a per-field weight -- it's a step function, and every mainstream platform
    # ships the same brackets. Per-bracket customization is a later refinement.


PPR = ScoringRules()
HALF_PPR = ScoringRules(reception=0.5)
STANDARD = ScoringRules(reception=0.0)

PRESETS = {"ppr": PPR, "half": HALF_PPR, "standard": STANDARD}


def rules_from(scoring: str | None = "ppr", custom: dict | None = None) -> ScoringRules:
    """Resolve a ScoringRules from a preset name and/or an explicit field dict.

    `custom` (a subset of ScoringRules field names -> values) wins when given --
    that's how an imported league's exact scoring (TE-premium, 6-pt pass TD, ...)
    flows through. Unknown keys are ignored so a platform's extra settings don't
    break the mapping.
    """
    if custom:
        fields = {f.name for f in dataclasses.fields(ScoringRules)}
        clean = {k: float(v) for k, v in custom.items() if k in fields}
        return ScoringRules(**clean)
    return PRESETS.get((scoring or "ppr").lower(), PPR)


def rules_to_dict(rules: ScoringRules) -> dict:
    """ScoringRules -> a JSON-serializable field dict (for persistence)."""
    return dataclasses.asdict(rules)


def preset_name(rules: ScoringRules) -> str:
    """The matching preset name ('ppr'/'half'/'standard'), or 'custom'."""
    for name, preset in PRESETS.items():
        if rules == preset:
            return name
    return "custom"


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
