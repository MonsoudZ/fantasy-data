"""Stacked ensemble -- the "colony": diverse specialist ants, combined by a queen.

A first attempt with three near-identical tree ants failed: they made the same
mistakes, so the queen had nothing to arbitrate (it split trust evenly and beat
nothing). Stacking only pays when the experts err *differently*. So this colony
forces diversity two ways:

  * feature-partitioned specialists -- each tree ant sees only ONE slice of the
    information and is blind to the rest, so they disagree by construction:
      - usage   ant: opportunity only (targets, carries, shares, snap share)
      - form    ant: production/efficiency only (trailing points, yards, EPA)
      - context ant: environment only (opponent defense, Vegas lines, injuries)
  * a different model class -- a linear (Ridge) ant over all features. Its
    smooth, extrapolating bias errs where trees don't.
  * an anchor -- the trailing-average baseline. Robust and low-variance.

The queen (a small LightGBM) sees the ants' out-of-fold predictions plus a
little context and learns whom to trust when: lean on usage for target hogs,
context in extreme game environments, the anchor for steady veterans.

Leakage rule (unchanged): the queen trains only on ant predictions made out of
sample, produced by a walk-forward pass where each ant trains strictly on
earlier weeks.

FINDING (2024 backtest): forcing diversity worked mechanically -- the queen now
beats every individual ant on MAE and no longer splits trust evenly (it leans
on the context ant most). But it only *ties* a single full-feature LightGBM,
winning marginally on MAE while losing marginally on RMSE and weekly rank. The
error-correlation report explains why: the strong ants stay 0.9+ correlated
because fantasy signal is concentrated in trailing production/usage, so every
competent model errs together; the only genuinely decorrelated ant (context)
is weak alone because it discards that dominant signal. Diversity and strength
were in tension. This module is therefore an EXPERIMENT and diagnostic, not the
production projector (that stays the single GBM in projections.py). It is the
scaffold to reuse once an ant is both strong AND different -- e.g. a neural
sequence model that reads a player's trajectory rather than a snapshot.

    from ffdata.ensemble import backtest_ensemble
    summary, trust, err_corr = backtest_ensemble()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features, feature_columns, INJURY_FEATURES
from .projections import TrailingAverageProjector, _weekly_spearman, _order_key

POSITIONS = ("QB", "RB", "WR", "TE")
_ANT_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=31,
                   min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                   random_state=0, verbose=-1, n_jobs=4)
_QUEEN_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15,
                     min_child_samples=30, random_state=0, verbose=-1, n_jobs=4)

# Feature partitions -- each specialist ant sees only its own slice.
_USAGE_STATS = ["targets", "receptions", "carries", "attempts",
                "target_share", "air_yards_share", "wopr", "racr", "snap_pct"]
_FORM_STATS = ["fp", "receiving_yards", "receiving_air_yards", "receiving_epa",
               "rushing_yards", "rushing_epa", "passing_yards", "passing_tds", "passing_epa"]
_WINDOWS = (3, 5)
USAGE_FEATURES = [f"{s}_r{n}" for n in _WINDOWS for s in _USAGE_STATS]
FORM_FEATURES = [f"{s}_r{n}" for n in _WINDOWS for s in _FORM_STATS]
CONTEXT_FEATURES = ([f"def_fp_allowed_r{n}" for n in _WINDOWS]
                    + ["team_implied_total", "opp_implied_total", "team_spread",
                       "game_total", "is_home"] + INJURY_FEATURES)

# name -> (model kind, feature slice). The anchor is handled separately.
_ANT_SPECS = [
    ("usage", "gbm", USAGE_FEATURES),
    ("form", "gbm", FORM_FEATURES),
    ("context", "gbm", CONTEXT_FEATURES),
    ("linear", "ridge", feature_columns()),
]
ANT_COLS = [f"ant_{name}" for name, _, _ in _ANT_SPECS] + ["ant_anchor"]
QUEEN_FEATURES = ANT_COLS + ["team_implied_total"] + [f"pos_{p}" for p in POSITIONS]


class Colony:
    """The diverse base experts: three feature specialists, a linear ant, an anchor."""

    def fit_ants(self, train: pd.DataFrame) -> dict:
        fitted = {}
        for name, kind, feats in _ANT_SPECS:
            if kind == "gbm":
                model = lgb.LGBMRegressor(**_ANT_PARAMS).fit(train[feats], train["fp"])
            else:  # ridge: impute + scale so the linear ant is well-behaved
                model = make_pipeline(SimpleImputer(strategy="median"),
                                      StandardScaler(), Ridge(alpha=10.0))
                model.fit(train[feats], train["fp"])
            fitted[name] = (model, feats)
        return fitted

    def predict_ants(self, fitted: dict, test: pd.DataFrame) -> pd.DataFrame:
        out = {f"ant_{name}": model.predict(test[feats]) for name, (model, feats) in fitted.items()}
        out["ant_anchor"] = TrailingAverageProjector().predict(test)
        return pd.DataFrame(out, index=test.index)


def generate_oof(feats: pd.DataFrame, oof_seasons: list[int], min_train: int = 1500) -> pd.DataFrame:
    """Walk-forward out-of-fold ant predictions -- the queen's training data."""
    colony = Colony()
    feats = feats.assign(_k=_order_key(feats))
    keys = sorted(feats.loc[feats["season"].isin(oof_seasons), "_k"].unique())
    chunks = []
    for k in keys:
        train = feats[feats["_k"] < k]
        test = feats[(feats["_k"] == k) & feats["fp_r3"].notna()]
        if len(train) < min_train or test.empty:
            continue
        preds = colony.predict_ants(colony.fit_ants(train), test)
        keep = test[["season", "week", "player_id", "position", "fp", "team_implied_total"]].copy()
        chunks.append(pd.concat([keep, preds], axis=1))
    oof = pd.concat(chunks, ignore_index=True)
    for p in POSITIONS:
        oof[f"pos_{p}"] = (oof["position"] == p).astype(int)
    oof["team_implied_total"] = oof["team_implied_total"].fillna(oof["team_implied_total"].median())
    return oof


def _metrics(df: pd.DataFrame, predcol: str) -> dict:
    err = df[predcol] - df["fp"]
    d = df.rename(columns={predcol: "pred"})
    return {
        "n": len(df),
        "MAE": round(float(err.abs().mean()), 3),
        "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
        "weekly_spearman": round(float(_weekly_spearman(d)), 4),
    }


def ant_correlation(oof: pd.DataFrame) -> pd.DataFrame:
    """How correlated the ants' *errors* are -- low is what makes stacking work."""
    errs = pd.DataFrame({c: oof[c] - oof["fp"] for c in ANT_COLS})
    return errs.corr().round(2)


def backtest_ensemble(
    train_from: int = 2019,
    oof_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare each ant against the stacked QUEEN on identical rows.

    Returns (summary, trust, err_corr).
    """
    oof_seasons = oof_seasons or [2023, 2024]
    test_seasons = test_seasons or [2024]
    seasons = list(range(train_from, max(oof_seasons) + 1))
    feats = build_features(seasons=seasons)

    oof = generate_oof(feats, oof_seasons)
    oof = oof.assign(_k=oof["season"] * 100 + oof["week"])

    res = []
    for k in sorted(oof.loc[oof["season"].isin(test_seasons), "_k"].unique()):
        qtrain, qtest = oof[oof["_k"] < k], oof[oof["_k"] == k]
        if len(qtrain) < 500 or qtest.empty:
            continue
        queen = lgb.LGBMRegressor(**_QUEEN_PARAMS).fit(qtrain[QUEEN_FEATURES], qtrain["fp"])
        res.append(qtest.assign(queen=queen.predict(qtest[QUEEN_FEATURES])))
    res = pd.concat(res, ignore_index=True)

    rows = [{"model": c, **_metrics(res, c)} for c in ANT_COLS]
    rows.append({"model": "QUEEN (ensemble)", **_metrics(res, "queen")})
    summary = pd.DataFrame(rows).set_index("model")

    qfinal = lgb.LGBMRegressor(**_QUEEN_PARAMS).fit(oof[QUEEN_FEATURES], oof["fp"])
    imp = pd.Series(qfinal.feature_importances_, index=QUEEN_FEATURES)[ANT_COLS]
    trust = (imp / imp.sum() * 100).round(1).rename("queen_trust_%").to_frame()
    return summary, trust, ant_correlation(res)


if __name__ == "__main__":
    pd.set_option("display.width", 100)
    summary, trust, err_corr = backtest_ensemble()
    print("Backtest 2024 -- each ant vs the stacked queen (identical rows):\n")
    print(summary)
    print("\nHow much the queen leans on each ant:\n")
    print(trust)
    print("\nAnt error correlations (lower = more diverse = better for stacking):\n")
    print(err_corr)
