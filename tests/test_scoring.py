"""Scoring is the source of truth for every downstream model -- pin it exactly."""

import pandas as pd

from ffdata.scoring import score, PPR, HALF_PPR, STANDARD, ScoringRules
from conftest import weekly_row


# A player line worth 34.0 PPR points, computed by hand:
#   passing_yards 300*.04=12, passing_tds 2*4=8, interceptions 1*-2=-2,
#   rushing_yards 20*.1=2, rushing_tds 1*6=6, receptions 5*1=5,
#   receiving_yards 50*.1=5, rushing_fumbles_lost 1*-2=-2  -> 34.0
STAT_LINE = dict(
    passing_yards=300, passing_tds=2, interceptions=1,
    rushing_yards=20, rushing_tds=1, receptions=5,
    receiving_yards=50, rushing_fumbles_lost=1,
)


def _one(rules, position="WR"):
    df = pd.DataFrame([weekly_row(position=position, **STAT_LINE)])
    return score(df, rules)["fp"].iloc[0]


def test_ppr_exact():
    assert _one(PPR) == 34.0


def test_half_ppr_drops_half_a_point_per_reception():
    assert _one(HALF_PPR) == 34.0 - 0.5 * 5  # 31.5


def test_standard_drops_all_reception_points():
    assert _one(STANDARD) == 34.0 - 1.0 * 5  # 29.0


def test_te_premium_only_applies_to_tight_ends():
    rules = ScoringRules(te_reception_bonus=0.5)
    assert _one(rules, position="TE") == 34.0 + 0.5 * 5  # 36.5
    assert _one(rules, position="WR") == 34.0            # unchanged


def test_missing_columns_default_to_zero():
    # A frame with only receptions present must not raise; other stats -> 0.
    df = pd.DataFrame([{"receptions": 3, "position": "RB"}])
    assert score(df, PPR)["fp"].iloc[0] == 3.0


def test_six_point_passing_td_config():
    rules = ScoringRules(pass_td=6.0)
    # base 34.0 with pass_td=4 for 2 TDs; at 6 pts that's +2 per TD = +4 -> 38.0
    assert _one(rules) == 38.0


def test_yardage_milestone_bonuses_fire_once_per_game():
    # STAT_LINE has exactly 300 passing yards -> the 300-yd bonus fires.
    assert _one(ScoringRules(bonus_pass_300=3.0)) == 34.0 + 3.0
    # ...but 20 rushing / 50 receiving yards are below their thresholds.
    assert _one(ScoringRules(bonus_rush_100=3.0)) == 34.0
    assert _one(ScoringRules(bonus_rec_100=3.0)) == 34.0
    # A genuine 100-yard game earns the bonus exactly once (not per yard).
    big = pd.DataFrame([weekly_row(position="RB", rushing_yards=120, receiving_yards=100)])
    got = score(big, ScoringRules(bonus_rush_100=3.0, bonus_rec_100=2.0))["fp"].iloc[0]
    assert got == 120 * 0.1 + 100 * 0.1 + 3.0 + 2.0    # 12 + 10 + 3 + 2 = 27
