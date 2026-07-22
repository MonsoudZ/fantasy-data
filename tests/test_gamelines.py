"""Game-line forecast assembly + leak-free team form (no data lake)."""

import numpy as np
import pandas as pd

from ffdata.gamelines import _forecast_rows, _team_form


def test_forecast_rows_leans_and_devig():
    test = pd.DataFrame({
        "home_team": ["KC", "SF"], "away_team": ["LV", "SEA"],
        "pred_total": [50.0, 40.0], "pred_margin": [7.0, -3.0],
        "total_line": [45.0, 44.0], "spread_line": [3.0, -1.0],
        "over_odds": [-110, -110], "under_odds": [-110, -110],
        "home_spread_odds": [-110, -110], "away_spread_odds": [-110, -110],
        "home_moneyline": [-150, 120], "away_moneyline": [130, -140],
    })
    r_tot = np.array([-3.0, 0.0, 3.0])
    r_mar = np.array([-7.0, 0.0, 7.0])
    rows = _forecast_rows(test, r_tot, r_mar).set_index("home")

    kc = rows.loc["KC"]
    assert kc["game"] == "LV @ KC"
    assert kc["total_lean"] == "over"          # 50 > 45
    assert kc["spread_lean"] == "KC"           # margin 7 > spread 3 -> home covers
    assert kc["ml_lean"] == "KC"               # P(margin>0) = 2/3 >= .5
    assert kc["model_over"] == 1.0             # need 45-50=-5; all residuals clear it
    assert kc["mkt_over"] == 0.5               # symmetric -110/-110

    sf = rows.loc["SF"]
    assert sf["total_lean"] == "under"         # 40 < 44
    assert sf["spread_lean"] == "SEA"          # margin -3 < spread -1 -> away
    assert sf["ml_lean"] == "SEA"              # P(margin>0) = 1/3 < .5
    # Market de-vig: home -150 / away +130 -> home fair ~0.58
    assert 0.55 < kc["mkt_home_win"] < 0.62


def test_team_form_is_trailing_and_leak_free():
    rows = []
    for wk, pf in enumerate([10, 20, 30, 40, 50], start=1):
        rows.append({"game_id": f"g{wk}", "season": 2024, "week": wk,
                     "home_team": "AA", "away_team": f"OPP{wk}",
                     "home_score": pf, "away_score": 0, "home_rest": 7, "away_rest": 7})
    form = _team_form(pd.DataFrame(rows))
    aa = form[(form.team == "AA")].sort_values("week")
    pf_form = aa["pf_form"].tolist()
    assert np.isnan(pf_form[0])                # no prior games
    assert pf_form[3] == 20.0                  # mean(10,20,30), excludes this week's 40
    assert pf_form[4] == 25.0                  # mean(10,20,30,40), excludes this week's 50
    # A team's own current-week points never appear in its form.
    assert 50.0 not in [v for v in pf_form if not np.isnan(v)]
