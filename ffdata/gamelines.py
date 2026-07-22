"""Game-line forecasts vs the market -- totals, spreads, moneylines.

Informational, not an edge play. The old edge finder measured a clean negative:
game betting markets are efficient to a public-data model, no edge survives the
vig. But putting our forecast next to the posted line still helps you decide --
predicted total vs the over/under, predicted margin vs the spread, model P(home
win) vs the moneyline. Lines AND results come straight from `schedules`, so
(unlike player props) there's nothing to paste.

    from ffdata.gamelines import game_forecasts
    game_forecasts(2024, 15)   # one row per game: our forecast beside the market

Leak-free: team-form features are trailing (shifted), models train only on games
before the week being forecast, and the probability scale comes from out-of-
sample residuals. NOTE: needs the `schedules` data ingested; the pure comparison
math (_forecast_rows) is unit-tested, the end-to-end path needs the lake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from .betting import _prob_over, american_to_prob
from .db import connect
from .gbm import gbm_params
from .ingest import FIRST_SEASON

FORM_WINDOW = 8
GAME_FEATURES = [
    "home_pf_form", "home_pa_form", "away_pf_form", "away_pa_form",
    "home_rest", "away_rest", "div_game", "dome", "neutral",
]
_PARAMS = gbm_params(n_estimators=300, num_leaves=15, min_child_samples=20)


def _team_form(games: pd.DataFrame) -> pd.DataFrame:
    """Per-team trailing points-for / points-against, shifted to exclude game W."""
    home = games.rename(columns={
        "home_team": "team", "home_score": "pf", "away_score": "pa", "home_rest": "rest"
    })[["game_id", "season", "week", "team", "pf", "pa", "rest"]].assign(is_home=1)
    away = games.rename(columns={
        "away_team": "team", "away_score": "pf", "home_score": "pa", "away_rest": "rest"
    })[["game_id", "season", "week", "team", "pf", "pa", "rest"]].assign(is_home=0)
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "season", "week"])
    grp = long.groupby("team", sort=False)
    long["pf_form"] = grp["pf"].transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=3).mean())
    long["pa_form"] = grp["pa"].transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=3).mean())
    return long


def build_games(train_from: int, through: int, con=None) -> pd.DataFrame:
    """One row per game in range with leak-free team-form features + lines + (if
    played) results. Unplayed games are kept so an upcoming week can be forecast --
    their form comes from prior played games via the shift."""
    con = con or connect()
    g = con.sql(f"""
        select game_id, season, week, home_team, away_team, home_score, away_score,
               result as home_margin, total as total_points, home_rest, away_rest,
               div_game, roof, location, spread_line, total_line,
               home_spread_odds, away_spread_odds, over_odds, under_odds,
               home_moneyline, away_moneyline
        from schedules
        where season between {train_from} and {through}
    """).df()

    form = _team_form(g)
    h = form[form.is_home == 1][["game_id", "pf_form", "pa_form"]].rename(
        columns={"pf_form": "home_pf_form", "pa_form": "home_pa_form"})
    a = form[form.is_home == 0][["game_id", "pf_form", "pa_form"]].rename(
        columns={"pf_form": "away_pf_form", "pa_form": "away_pa_form"})
    g = g.merge(h, on="game_id").merge(a, on="game_id")

    g["dome"] = g["roof"].isin(["dome", "closed"]).astype(int)
    g["neutral"] = (g["location"] != "Home").astype(int)
    g = g.dropna(subset=["home_pf_form", "away_pf_form"])
    return g.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def _walk_forward_resid(games: pd.DataFrame, target: str, seasons: list[int],
                        min_train: int = 200) -> np.ndarray:
    """Out-of-sample residuals: retrain before each week in `seasons`, predict it."""
    g = games[games[target].notna()].assign(_k=lambda d: d.season * 100 + d.week)
    chunks = []
    for k in sorted(g.loc[g.season.isin(seasons), "_k"].unique()):
        train, test = g[g._k < k], g[g._k == k]
        if len(train) < min_train or test.empty:
            continue
        model = lgb.LGBMRegressor(**_PARAMS).fit(train[GAME_FEATURES], train[target])
        chunks.append(test[target].to_numpy() - model.predict(test[GAME_FEATURES]))
    return np.concatenate(chunks) if chunks else np.array([])


def _forecast_rows(test: pd.DataFrame, r_tot: np.ndarray, r_mar: np.ndarray) -> pd.DataFrame:
    """Assemble the model-vs-market comparison for a week's games.

    `test` carries pred_total/pred_margin, the posted lines, and the odds columns;
    r_tot/r_mar are the models' OOS residual pools for the P(over) scale.
    """
    t = test.reset_index(drop=True)
    pt, pm = t["pred_total"].to_numpy(), t["pred_margin"].to_numpy()
    tline, sline = t["total_line"].to_numpy(), t["spread_line"].to_numpy()

    model_over = _prob_over(r_tot, pt, tline)
    model_cover = _prob_over(r_mar, pm, sline)
    model_home = _prob_over(r_mar, pm, np.zeros(len(t)))

    def fair(a, b):
        pa, pb = american_to_prob(t[a]), american_to_prob(t[b])
        return pa / (pa + pb)
    mkt_over = fair("over_odds", "under_odds")
    mkt_cover = fair("home_spread_odds", "away_spread_odds")
    mkt_home = fair("home_moneyline", "away_moneyline")

    rows = []
    for i, g in t.iterrows():
        rows.append({
            "game": f"{g['away_team']} @ {g['home_team']}",
            "home": g["home_team"], "away": g["away_team"],
            "total_line": round(float(tline[i]), 1), "pred_total": round(float(pt[i]), 1),
            "total_lean": "over" if pt[i] > tline[i] else "under",
            "model_over": round(float(model_over[i]), 3), "mkt_over": round(float(mkt_over[i]), 3),
            "spread_line": round(float(sline[i]), 1), "pred_margin": round(float(pm[i]), 1),
            "spread_lean": g["home_team"] if pm[i] > sline[i] else g["away_team"],
            "model_home_cover": round(float(model_cover[i]), 3),
            "mkt_home_cover": round(float(mkt_cover[i]), 3),
            "model_home_win": round(float(model_home[i]), 3),
            "mkt_home_win": round(float(mkt_home[i]), 3),
            "ml_lean": g["home_team"] if model_home[i] >= 0.5 else g["away_team"],
        })
    return pd.DataFrame(rows)


def game_forecasts(season: int, week: int, train_from: int = FIRST_SEASON, con=None) -> pd.DataFrame:
    """One row per game in (season, week): our forecast beside the market line."""
    con = con or connect()
    games = build_games(train_from, season, con=con).assign(_k=lambda d: d.season * 100 + d.week)
    k = season * 100 + week
    train = games[(games._k < k) & games["total_points"].notna()]
    test = games[games._k == k].copy()
    if train.empty or test.empty:
        return pd.DataFrame()

    # Probability scale: OOS residuals from the last two seasons present in train.
    resid_seasons = sorted(train["season"].unique())[-2:]
    r_tot = _walk_forward_resid(games, "total_points", resid_seasons)
    r_mar = _walk_forward_resid(games, "home_margin", resid_seasons)
    if not len(r_tot) or not len(r_mar):
        return pd.DataFrame()

    test["pred_total"] = lgb.LGBMRegressor(**_PARAMS).fit(
        train[GAME_FEATURES], train["total_points"]).predict(test[GAME_FEATURES])
    test["pred_margin"] = lgb.LGBMRegressor(**_PARAMS).fit(
        train[GAME_FEATURES], train["home_margin"]).predict(test[GAME_FEATURES])
    return _forecast_rows(test, r_tot, r_mar)
