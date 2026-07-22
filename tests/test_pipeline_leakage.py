"""End-to-end temporal-correctness tests for the assembled pipeline.

The existing test_features.py checks the leakage *helpers* in isolation. These
tests close the gap flagged in the audit: the two places where leakage would be
most damaging and were previously unguarded --

  * walk_forward (the backtest protocol behind every reported metric), and
  * build_features assembled end to end (all merges, via a synthetic DuckDB).

Both run in CI without a data lake: walk_forward on a synthetic frame, and
build_features against tiny in-memory DuckDB tables whose ground truth we
control exactly.
"""

import duckdb
import numpy as np
import pandas as pd

from ffdata.features import USAGE_COLS, build_features
from ffdata.projections import walk_forward


# --------------------------------------------------------------------------- #
# walk_forward: training set must be strictly before the week being scored
# --------------------------------------------------------------------------- #

def test_walk_forward_trains_strictly_on_the_past():
    rows = []
    for season in (2021, 2022, 2023):
        for week in range(1, 4):
            for pid in range(6):
                rows.append({"season": season, "week": week, "player_id": f"p{pid}",
                             "position": "WR", "fp": float(pid + week), "fp_r3": 5.0})
    feats = pd.DataFrame(rows)

    retrains = []

    class SpyProjector:
        name = "spy"

        def fit(self, train):
            # Record the latest chronological key present in the training set.
            self._train_max_k = int(train["_k"].max())
            retrains.append(self._train_max_k)
            return self

        def predict(self, test):
            # The core guarantee: every training row is strictly BEFORE the
            # week being predicted. A leak (train containing _k >= k) trips this.
            assert int(test["_k"].min()) > self._train_max_k
            return np.zeros(len(test))

    preds = walk_forward(feats, SpyProjector(), test_seasons=[2023], min_train_rows=1)

    # Only the requested test season is scored...
    assert set(preds["season"].unique()) == {2023}
    # ...and the model retrained once per distinct test week (expanding window).
    assert len(retrains) == 3
    # Retrain cutoffs strictly increase as the window expands.
    assert retrains == sorted(retrains) and len(set(retrains)) == 3


def test_walk_forward_respects_min_train_rows():
    rows = [{"season": 2023, "week": w, "player_id": f"p{p}", "position": "WR",
             "fp": 1.0, "fp_r3": 1.0} for w in range(1, 4) for p in range(3)]
    feats = pd.DataFrame(rows)

    class Proj:
        name = "p"

        def fit(self, train):
            return self

        def predict(self, test):
            return np.zeros(len(test))

    # With a floor above the available history, no week has enough train rows.
    preds = walk_forward(feats, Proj(), test_seasons=[2023], min_train_rows=10_000)
    assert preds.empty


# --------------------------------------------------------------------------- #
# build_features: leak-free trailing features through the whole pipeline
# --------------------------------------------------------------------------- #

def _dummy(**cols) -> pd.DataFrame:
    """One typed, non-matching row so DuckDB infers column types (empty frames
    lose dtypes and break joins)."""
    return pd.DataFrame([cols])


def _con_with_weekly(weekly: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("weekly", weekly)
    # Auxiliary tables are left-merged; give each a single non-matching row so
    # the merges add their columns (all NaN) without matching our player.
    con.register("rosters", _dummy(season=2000, pfr_id="none", gsis_id="none"))
    con.register("snap_counts",
                 _dummy(season=2000, week=1, pfr_player_id="none", offense_pct=0.0))
    con.register("injuries",
                 _dummy(gsis_id="none", season=2000, week=1,
                        report_status="ACT", practice_status="Full"))
    con.register("schedules",
                 _dummy(season=2000, week=1, home_team="AAA", away_team="BBB",
                        spread_line=0.0, total_line=40.0))
    return con


def _one_player_weekly() -> pd.DataFrame:
    """One WR, 2023 weeks 1-4, receiving_yards -> PPR fp of 10/20/30/40."""
    base = {c: 0.0 for c in USAGE_COLS if c not in ("fp", "snap_pct")}
    rows = []
    for wk, ry in enumerate([100.0, 200.0, 300.0, 400.0], start=1):
        row = dict(base)
        row.update({"player_id": "P1", "season": 2023, "week": wk, "position": "WR",
                    "recent_team": "KC", "opponent_team": "LV", "season_type": "REG",
                    "receiving_yards": ry})
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_features_fp_target_and_trailing_are_leak_free():
    con = _con_with_weekly(_one_player_weekly())
    feats = build_features(seasons=[2023], con=con).sort_values("week").reset_index(drop=True)

    # score() reproduced the PPR target from raw receiving yards.
    assert list(feats["fp"]) == [10.0, 20.0, 30.0, 40.0]

    r3 = feats["fp_r3"].tolist()
    assert np.isnan(r3[0])        # week 1: no prior games -> undefined
    assert r3[1] == 10.0          # week 2: mean of week 1 only
    assert r3[2] == 15.0          # week 3: mean(10, 20)
    assert r3[3] == 20.0          # week 4: mean(10, 20, 30) -- excludes week 4

    # The cardinal property: a week's own points are NEVER in its trailing feature.
    assert (feats["fp_r3"].fillna(-1.0) != feats["fp"]).all()


def test_build_features_rolling_does_not_leak_across_a_gap():
    # Same player, but with an r5 window we can also check the wider trailing mean.
    con = _con_with_weekly(_one_player_weekly())
    feats = build_features(seasons=[2023], con=con).sort_values("week").reset_index(drop=True)
    r5 = feats["fp_r5"].tolist()
    assert np.isnan(r5[0])
    assert r5[3] == 20.0          # mean(10, 20, 30); window 5 but only 3 priors
