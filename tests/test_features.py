"""The feature layer's whole value is being leak-free -- prove it on toy data."""

import numpy as np
import pandas as pd

from ffdata.features import _rolling_usage, _opponent_defense, _implied_totals
from conftest import weekly_row


def test_rolling_usage_is_a_trailing_mean_excluding_current_week(one_player_weekly):
    out = _rolling_usage(one_player_weekly, windows=(2,)).sort_values("week")
    # fp is 10,20,30,40; fp_r2 must use only PRIOR weeks.
    assert np.isnan(out["fp_r2"].iloc[0])          # debut: no history
    assert out["fp_r2"].iloc[1] == 10.0            # mean(10)
    assert out["fp_r2"].iloc[2] == 15.0            # mean(10,20)
    assert out["fp_r2"].iloc[3] == 25.0            # mean(20,30)


def test_rolling_never_peeks_at_the_current_week(one_player_weekly):
    out = _rolling_usage(one_player_weekly, windows=(3,))
    # If any rolled value equaled its own week's fp, that'd be leakage.
    assert not (out["fp_r3"].fillna(-1) == out["fp"]).any()


def test_implied_totals_split_the_line_by_the_spread():
    sched = pd.DataFrame([{
        "season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
        "spread_line": 3.0, "total_line": 46.0,
    }])
    imp = _implied_totals(sched).set_index("team")
    # total/2 +/- spread/2  ->  home 24.5, away 21.5
    assert imp.loc["KC", "team_implied_total"] == 24.5
    assert imp.loc["KC", "opp_implied_total"] == 21.5
    assert imp.loc["BAL", "team_implied_total"] == 21.5
    assert imp.loc["KC", "is_home"] == 1 and imp.loc["BAL", "is_home"] == 0


def test_implied_totals_sum_to_the_game_total():
    sched = pd.DataFrame([{
        "season": 2024, "week": 1, "home_team": "A", "away_team": "B",
        "spread_line": -6.5, "total_line": 51.0,
    }])
    imp = _implied_totals(sched)
    assert imp["team_implied_total"].sum() == 51.0


def test_opponent_defense_trails_points_allowed_leak_free():
    # Defense 'LV' faces WRs who score 20 (wk1) then 30 (wk2), then a wk3 game.
    rows = [
        weekly_row(week=1, position="WR", opponent_team="LV", fp=20.0),
        weekly_row(week=2, position="WR", opponent_team="LV", fp=30.0),
        weekly_row(week=3, position="WR", opponent_team="LV", fp=99.0),
    ]
    d = _opponent_defense(pd.DataFrame(rows), windows=(2,)).sort_values("week")
    col = "def_fp_allowed_r2"
    assert np.isnan(d[col].iloc[0])       # wk1: no prior games
    assert d[col].iloc[1] == 20.0         # entering wk2: allowed 20 in wk1
    assert d[col].iloc[2] == 25.0         # entering wk3: mean(20,30) -- wk3 unseen
