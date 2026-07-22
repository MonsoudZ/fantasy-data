"""Draft/dynasty season aggregation honors the league's ScoringRules.

Previously draft/dynasty summed the precomputed `fantasy_points_ppr` column, so
every league got PPR values. These tests prove the season totals now flow from
scoring.score() over raw stats, so half-PPR and standard leagues differ correctly
-- run in CI on a synthetic in-memory DuckDB (no data lake).
"""

import duckdb
import pandas as pd

from ffdata.draft import _season_agg
from ffdata.scoring import HALF_PPR, PPR, STANDARD

# Aggregation columns _season_agg reads (beyond the scoring inputs).
_AGG_COLS = ["targets", "carries", "receptions", "receiving_yards", "rushing_yards",
             "passing_yards", "passing_tds", "rushing_tds", "receiving_tds", "target_share"]


def _weekly_reception_heavy() -> pd.DataFrame:
    """One WR, two REG games, 5 receptions + 50 receiving yards each."""
    base = {c: 0.0 for c in _AGG_COLS}
    rows = []
    for wk in (1, 2):
        row = dict(base)
        row.update({"player_id": "W1", "season": 2023, "position": "WR",
                    "player_display_name": "Rec Guy", "season_type": "REG",
                    "receptions": 5.0, "receiving_yards": 50.0, "target_share": 0.2})
        rows.append(row)
    return pd.DataFrame(rows)


def _con(weekly: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("weekly", weekly)
    return con


def _fp(con, rules) -> float:
    agg = _season_agg(con, rules).set_index("player_id")
    return float(agg.loc["W1", "fp"])


def test_season_totals_track_the_scoring_rules():
    con = _con(_weekly_reception_heavy())
    ppr, half, std = _fp(con, PPR), _fp(con, HALF_PPR), _fp(con, STANDARD)

    # Receiving yards are common to all three (2 games * 50 * 0.1 = 10 pts).
    # Receptions: 10 total -> PPR +10, half +5, standard +0 over the yardage base.
    assert std == 10.0
    assert half == 15.0
    assert ppr == 20.0
    # The whole point: reception-scoring changes the value, so leagues differ.
    assert ppr > half > std


def test_default_rules_are_ppr():
    con = _con(_weekly_reception_heavy())
    agg_default = _season_agg(con).set_index("player_id")
    assert float(agg_default.loc["W1", "fp"]) == 20.0
