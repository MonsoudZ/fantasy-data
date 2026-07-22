"""Projection models + walk-forward backtest.

Roadmap step 3. Two projectors predict a player's fantasy points for an
upcoming week from the leak-free feature layer:

  * TrailingAverageProjector -- the baseline. A blend of a player's recent
                                scoring (fp_r3 / fp_r5). Hard to beat, cheap.
  * GBMProjector             -- LightGBM over the full feature set.

Both run through the SAME walk-forward protocol: to score week W we train only
on weeks strictly before W, so the backtest never sees the future. Metrics
report accuracy (MAE / RMSE) and -- what actually matters for start/sit and
lineups -- weekly rank quality (Spearman): ordering players right beats nailing
their absolute point totals.

    from ffdata.projections import backtest
    print(backtest(train_from=2019, test_seasons=[2023, 2024]))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from .features import build_features, feature_columns
from .gbm import gbm_params


def _order_key(df: pd.DataFrame) -> pd.Series:
    """Monotonic chronological key; weeks reset each season, so fold season in."""
    return df["season"] * 100 + df["week"]


class TrailingAverageProjector:
    """Baseline: a fixed blend of a player's trailing fantasy-point averages."""

    name = "trailing_avg"

    def __init__(self, w3: float = 0.6, w5: float = 0.4):
        self.w3, self.w5 = w3, w5

    def fit(self, train: pd.DataFrame) -> "TrailingAverageProjector":
        return self  # nothing to learn -- the features already encode it

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        r3 = test["fp_r3"].fillna(test["fp_r5"])
        r5 = test["fp_r5"].fillna(test["fp_r3"])
        return (self.w3 * r3 + self.w5 * r5).to_numpy()


class GBMProjector:
    """LightGBM gradient boosting over the full feature layer."""

    name = "lightgbm"

    def __init__(self, features: list[str] | None = None, **params):
        self.features = features or feature_columns()
        self.params = gbm_params(n_estimators=400, num_leaves=31, min_child_samples=40)
        self.params.update(params)
        self.model: lgb.LGBMRegressor | None = None

    def fit(self, train: pd.DataFrame) -> "GBMProjector":
        self.model = lgb.LGBMRegressor(**self.params)
        # LightGBM consumes NaNs natively -- early-career rows stay usable.
        self.model.fit(train[self.features], train["fp"])
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return self.model.predict(test[self.features])


def walk_forward(
    feats: pd.DataFrame,
    projector,
    test_seasons: list[int],
    eval_col: str = "fp_r3",
    min_train_rows: int = 1000,
) -> pd.DataFrame:
    """Expanding-window backtest: retrain before each test week, predict it.

    Only rows where `eval_col` is defined are scored, so every model is judged
    on the same comparable set (players with enough history to have a baseline).
    """
    feats = feats.assign(_k=_order_key(feats))
    test_keys = sorted(feats.loc[feats["season"].isin(test_seasons), "_k"].unique())
    chunks = []
    for k in test_keys:
        train = feats[feats["_k"] < k]
        if len(train) < min_train_rows:
            continue
        test = feats[(feats["_k"] == k) & feats[eval_col].notna()]
        if test.empty:
            continue
        projector.fit(train)
        preds = projector.predict(test)
        chunks.append(
            test[["season", "week", "player_id", "position", "fp"]].assign(pred=preds)
        )
    if not chunks:  # no test week had enough history -- return empty, don't crash
        return pd.DataFrame(columns=["season", "week", "player_id", "position", "fp", "pred"])
    return pd.concat(chunks, ignore_index=True)


def _weekly_spearman(pred_df: pd.DataFrame) -> float:
    """Mean per-week rank correlation between projection and actual points."""
    def one(g: pd.DataFrame) -> float:
        if g["fp"].nunique() < 2 or g["pred"].nunique() < 2:
            return np.nan
        return spearmanr(g["pred"], g["fp"]).correlation
    return pred_df.groupby(["season", "week"], group_keys=False).apply(one).mean()


def evaluate(pred_df: pd.DataFrame) -> dict:
    """Accuracy + rank metrics for a backtest's predictions."""
    err = pred_df["pred"] - pred_df["fp"]
    return {
        "n": len(pred_df),
        "MAE": round(mean_absolute_error(pred_df["fp"], pred_df["pred"]), 3),
        "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
        "weekly_spearman": round(float(_weekly_spearman(pred_df)), 4),
    }


def backtest(
    train_from: int = 2019,
    test_seasons: list[int] | None = None,
    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE"),
) -> pd.DataFrame:
    """Build features, backtest baseline vs LightGBM, return a metrics table."""
    test_seasons = test_seasons or [2023, 2024]
    seasons = list(range(train_from, max(test_seasons) + 1))
    feats = build_features(seasons=seasons, positions=positions)

    rows = []
    for proj in (TrailingAverageProjector(), GBMProjector()):
        preds = walk_forward(feats, proj, test_seasons)
        rows.append({"model": proj.name, **evaluate(preds)})
    return pd.DataFrame(rows).set_index("model")


if __name__ == "__main__":
    pd.set_option("display.width", 100)
    print(backtest())
