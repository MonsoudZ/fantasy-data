"""Edge finder: model probability vs the market's implied odds.

Roadmap step 5 -- the payoff. Steps 2-4 built the machinery on *player* fantasy
points; the betting markets in `schedules`, though, price *game* outcomes
(totals, spreads, moneylines). So this module applies the same discipline --
leak-free features, walk-forward training, empirical residual distributions --
to two game-outcome models:

  * total points  -> P(over the posted total line)
  * home margin   -> P(home covers the spread), P(home wins outright)

Each model turns a point prediction into a probability by asking what fraction
of its own out-of-sample residuals would clear the line. That model probability
is compared to the market's *de-vigged* implied probability; the gap is the
edge. Then -- and this is the only test that matters -- we bet every flagged
edge at the actual odds offered and track the running profit. An edge that
doesn't survive the vig isn't an edge.

    from ffdata.edge import find_edges
    summary, bets = find_edges(train_from=2019, test_seasons=[2023, 2024])
    print(summary)   # ROI / record per market -- the honest scorecard

Model inputs are team form + schedule context only; the game's own betting line
is never a feature (that would just relearn the market). Player projections
would feed a *props* edge finder instead -- but nflverse ships no prop lines,
so that waits for a prop-odds source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from .db import connect

# Team-form lookback (games) for the rolling offense/defense features.
FORM_WINDOW = 8
GAME_FEATURES = [
    "home_pf_form", "home_pa_form", "away_pf_form", "away_pa_form",
    "home_rest", "away_rest", "div_game", "dome", "neutral",
]


def american_to_prob(odds: pd.Series | np.ndarray) -> np.ndarray:
    """Convert American odds to their raw (vig-inclusive) implied probability."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # np.where evaluates both branches; the unused one can divide by zero.
        return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def american_profit(odds: float, won: bool) -> float:
    """Profit on a 1-unit stake at American `odds` (push handled by caller)."""
    if not won:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / -odds


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
    """Assemble one row per played game with leak-free features, targets, and lines."""
    con = con or connect()
    g = con.sql(f"""
        select game_id, season, week, home_team, away_team, home_score, away_score,
               result as home_margin, total as total_points, home_rest, away_rest,
               div_game, roof, location, spread_line, total_line,
               home_spread_odds, away_spread_odds, over_odds, under_odds,
               home_moneyline, away_moneyline
        from schedules
        where season between {train_from} and {through} and result is not null
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


def _walk_forward_predict(games: pd.DataFrame, target: str, test_seasons: list[int]) -> pd.DataFrame:
    """Retrain a GBM before each test week; return test rows with `pred` and residual."""
    key = games["season"] * 100 + games["week"]
    games = games.assign(_k=key)
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=15,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  random_state=0, verbose=-1)
    chunks = []
    for k in sorted(games.loc[games.season.isin(test_seasons), "_k"].unique()):
        train = games[games._k < k]
        test = games[games._k == k]
        if len(train) < 200:
            continue
        model = lgb.LGBMRegressor(**params)
        model.fit(train[GAME_FEATURES], train[target])
        test = test.assign(pred=model.predict(test[GAME_FEATURES]))
        chunks.append(test)
    out = pd.concat(chunks, ignore_index=True)
    out["resid"] = out[target] - out["pred"]
    return out


def _prob_over(resid: np.ndarray, pred: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Empirical P(outcome > line) = share of residuals clearing (line - pred)."""
    need = (line - pred)[:, None]        # (games, 1)
    return (resid[None, :] > need).mean(axis=1)


def find_edges(
    train_from: int = 2019,
    test_seasons: list[int] | None = None,
    threshold: float = 0.05,
    con=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest model-vs-market edges across totals, spreads, and moneylines.

    Returns (summary, bets): `summary` is per-market ROI/record; `bets` is the
    chronological wager log (with cumulative units) for tracking over time.
    """
    test_seasons = test_seasons or [2023, 2024]
    games = build_games(train_from, max(test_seasons), con=con)

    tot = _walk_forward_predict(games, "total_points", test_seasons)
    mar = _walk_forward_predict(games, "home_margin", test_seasons)
    m = tot[["game_id", "season", "week", "pred", "total_points"]].rename(
        columns={"pred": "pred_total"}
    ).merge(
        mar[["game_id", "pred", "home_margin"]].rename(columns={"pred": "pred_margin"}),
        on="game_id",
    ).merge(
        games[["game_id", "spread_line", "total_line", "home_spread_odds",
               "away_spread_odds", "over_odds", "under_odds",
               "home_moneyline", "away_moneyline"]],
        on="game_id",
    ).sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    tot_resid = tot["resid"].to_numpy()
    mar_resid = mar["resid"].to_numpy()

    # Model probabilities for the three "yes" outcomes.
    m["p_over"] = _prob_over(tot_resid, m["pred_total"].to_numpy(), m["total_line"].to_numpy())
    m["p_home_cover"] = _prob_over(mar_resid, m["pred_margin"].to_numpy(), m["spread_line"].to_numpy())
    m["p_home_win"] = _prob_over(mar_resid, m["pred_margin"].to_numpy(), np.zeros(len(m)))

    # Market de-vigged fair probabilities.
    def fair(a, b):
        pa, pb = american_to_prob(m[a]), american_to_prob(m[b])
        return pa / (pa + pb)
    m["fair_over"] = fair("over_odds", "under_odds")
    m["fair_home_cover"] = fair("home_spread_odds", "away_spread_odds")
    m["fair_home_win"] = fair("home_moneyline", "away_moneyline")

    markets = [
        # name, model p(yes), market fair p(yes), odds if bet yes / no, win test
        ("total", "p_over", "fair_over", "over_odds", "under_odds",
         lambda r: r.total_points > r.total_line, lambda r: r.total_points < r.total_line),
        ("spread", "p_home_cover", "fair_home_cover", "home_spread_odds", "away_spread_odds",
         lambda r: r.home_margin > r.spread_line, lambda r: r.home_margin < r.spread_line),
        ("moneyline", "p_home_win", "fair_home_win", "home_moneyline", "away_moneyline",
         lambda r: r.home_margin > 0, lambda r: r.home_margin < 0),
    ]

    bets = []
    for name, pcol, faircol, yes_odds, no_odds, yes_win, no_win in markets:
        for _, r in m.iterrows():
            e = r[pcol] - r[faircol]
            if e > threshold:      # model likes the "yes" side
                side, odds, won = "yes", r[yes_odds], yes_win(r)
            elif -e > threshold:   # model likes the "no" side
                side, odds, won = "no", r[no_odds], no_win(r)
            else:
                continue
            push = not yes_win(r) and not no_win(r)  # exact line hit
            profit = 0.0 if push else american_profit(float(odds), bool(won))
            bets.append({
                "season": r.season, "week": r.week, "game_id": r.game_id,
                "market": name, "side": side, "edge": round(float(e), 3),
                "odds": float(odds), "won": bool(won), "push": push, "profit": profit,
            })

    bets = pd.DataFrame(bets).sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    if not bets.empty:
        bets["cum_units"] = bets.groupby("market")["profit"].cumsum()

    rows = []
    for name, gdf in bets.groupby("market"):
        graded = gdf[~gdf.push]
        rows.append({
            "market": name, "bets": len(gdf), "wins": int(graded.won.sum()),
            "pushes": int(gdf.push.sum()),
            "win_pct": round(graded.won.mean() * 100, 1) if len(graded) else np.nan,
            "units": round(gdf.profit.sum(), 2),
            "roi_pct": round(gdf.profit.sum() / len(gdf) * 100, 1) if len(gdf) else np.nan,
        })
    summary = pd.DataFrame(rows).set_index("market") if rows else pd.DataFrame()
    return summary, bets


if __name__ == "__main__":
    pd.set_option("display.width", 100)
    summary, bets = find_edges()
    n_games = bets["game_id"].nunique() if not bets.empty else 0
    print(f"Backtest 2023-2024 | {len(bets)} bets over {n_games} games "
          f"| break-even win% ~52.4 (at -110)\n")
    print(summary)
