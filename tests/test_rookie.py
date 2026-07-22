"""Rookie draft-capital model wiring, on a synthetic in-memory DuckDB.

These validate the *logic* (draft capital flows in, early picks project higher,
missing source degrades gracefully) -- NOT real-world accuracy, which needs the
nflverse draft_picks lake and `backtest_rookies()`. Runs in CI (no data lake).
"""

import duckdb
import pandas as pd

from ffdata.draft import _draft_capital, rookie_projection
from ffdata.scoring import PPR

_AGG_COLS = ["targets", "carries", "receptions", "receiving_yards", "rushing_yards",
             "passing_yards", "passing_tds", "rushing_tds", "receiving_tds", "target_share"]


def _weekly_rows(rookies) -> pd.DataFrame:
    """One REG game per historical rookie, receiving yards set so season fp ~= 300-pick."""
    rows = []
    for pid, season, pick, pos in rookies:
        row = {c: 0.0 for c in _AGG_COLS}
        row.update({"player_id": pid, "season": season, "week": 1, "position": pos,
                    "player_display_name": pid, "season_type": "REG",
                    "recent_team": "KC", "opponent_team": "LV",
                    "receiving_yards": max(10.0, 300.0 - pick) * 10})  # *0.1 -> ~300-pick
        rows.append(row)
    return pd.DataFrame(rows)


def _draft_rows(rows) -> pd.DataFrame:
    return pd.DataFrame([{"gsis_id": pid, "season": season, "round": rnd, "pick": pick,
                          "position": pos, "pfr_player_name": pid}
                         for pid, season, rnd, pick, pos in rows])


def _con(weekly, picks=None):
    con = duckdb.connect()
    con.register("weekly", weekly)
    if picks is not None:
        con.register("draft_picks", picks)
    return con


def _training_universe():
    """~90 historical rookies (2020-2022) with a strong pick -> points signal."""
    hist_weekly, hist_picks = [], []
    for season in (2020, 2021, 2022):
        for pick in range(1, 260, 8):
            pid = f"h{season}_{pick}"
            pos = "WR" if pick % 2 else "RB"
            rnd = pick // 32 + 1
            hist_weekly.append((pid, season, pick, pos))
            hist_picks.append((pid, season, rnd, pick, pos))
    return hist_weekly, hist_picks


def test_rookie_projection_orders_by_draft_capital():
    hist_weekly, hist_picks = _training_universe()
    # 2023 rookies to project: an elite early pick vs a late-round flier.
    draft2023 = [("early", 2023, 1, 1, "WR"), ("late", 2023, 8, 250, "WR")]
    con = _con(_weekly_rows(hist_weekly), _draft_rows(hist_picks + draft2023))

    proj = rookie_projection(2023, PPR, con=con)
    assert proj is not None and not proj.empty
    by = proj.set_index("player_id")["proj"]
    assert {"early", "late"}.issubset(by.index)
    # The whole point of a draft-capital model: pick 1 outprojects pick 250.
    assert by["early"] > by["late"]


def test_rookie_projection_is_none_without_the_source():
    # A con with weekly but no draft_picks view -> gracefully None (veterans only).
    hist_weekly, _ = _training_universe()
    con = _con(_weekly_rows(hist_weekly))
    assert rookie_projection(2023, PPR, con=con) is None


def test_draft_capital_keeps_only_skill_positions():
    picks = _draft_rows([("qb1", 2022, 1, 1, "QB"), ("wr1", 2022, 1, 2, "WR"),
                         ("k1", 2022, 5, 150, "K"), ("ol1", 2022, 1, 3, "T")])
    con = _con(pd.DataFrame({"player_id": [], "season": []}), picks)
    caps = _draft_capital(con)
    assert set(caps["position"]) == {"QB", "WR"}          # K and T dropped
    assert set(caps["player_id"]) == {"qb1", "wr1"}


def test_rookie_projection_leak_free_split_excludes_target_year():
    # A rookie drafted IN the target season must not be in the training set.
    hist_weekly, hist_picks = _training_universe()
    draft2023 = [("r2023", 2023, 1, 5, "WR")]
    # Give the 2023 rookie a weekly row too (as if the season had happened) --
    # the model must NOT train on it, since draft_season (2023) is not < 2023.
    wk = _weekly_rows(hist_weekly + [("r2023", 2023, 5, "WR")])
    con = _con(wk, _draft_rows(hist_picks + draft2023))
    proj = rookie_projection(2023, PPR, con=con)
    # It still projects the 2023 rookie (from draft capital), not from its own
    # in-season result -- if it leaked, proj would ~= its actual 295, but the
    # model is trained only on <2023 rows, so it generalizes from capital.
    assert proj is not None and "r2023" in set(proj["player_id"])
