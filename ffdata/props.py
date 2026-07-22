"""Player-props edge finder: our projections vs sportsbook prop lines.

Game betting lines proved efficient to a public-data model (no edge survived the
vig). Player props are softer -- more markets, lower limits, less sharp attention
-- so a competent per-stat projection has a real chance to beat them. This is the
one place the projection stack might find money.

Props are priced on *individual stats* (receiving yards, receptions, passing
yards, ...), not fantasy points, so each market gets its own model: the same
leak-free feature layer and empirical-residual machinery as everywhere else,
with the target column swapped. P(over the line) is the share of the model's
out-of-sample residuals that would clear it (empirical-residual tail, see
`ffdata.betting._prob_over`), and validated calibrated per market (see `calibrate`).

Data note: nflverse ships no prop odds. You bring the lines (a CSV: player,
market, line, over_odds, under_odds); this prices them. The engine and its
calibration are the deliverable; the odds are your input.

    from ffdata.props import price_props
    edges = price_props(prop_df, season=2024, week=15)   # +EV bets, ranked
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from .features import build_features, feature_columns
from .betting import american_to_prob, american_profit, _prob_over
from .gbm import gbm_params
from .ingest import FIRST_SEASON
from .optimize import _norm

# Prop market -> the positions that accrue that stat.
MARKETS = {
    "passing_yards": ("QB",),
    "passing_tds": ("QB",),
    "receiving_yards": ("WR", "TE", "RB"),
    "receptions": ("WR", "TE", "RB"),
    "rushing_yards": ("RB", "QB", "WR"),
}
_PARAMS = gbm_params(n_estimators=300, num_leaves=31, min_child_samples=40)


def _ev_side(p_over: float, over_odds: float, under_odds: float) -> tuple[str, float]:
    """Pick the higher-EV side of a prop and its EV per 1u at the offered odds."""
    ev_over = p_over * american_profit(over_odds, True) - (1 - p_over)
    ev_under = (1 - p_over) * american_profit(under_odds, True) - p_over
    return ("over", ev_over) if ev_over >= ev_under else ("under", ev_under)


def _rows(feats: pd.DataFrame, positions) -> pd.DataFrame:
    f = feats[feats["position"].isin(positions)].copy()
    f["_k"] = f["season"] * 100 + f["week"]
    return f


def _oos_residuals(feats: pd.DataFrame, stat: str, positions, resid_season: int) -> pd.DataFrame:
    """Train on seasons < resid_season, predict it -> out-of-sample residuals."""
    f = _rows(feats, positions)
    cols = feature_columns()
    train = f[f["season"] < resid_season]
    test = f[(f["season"] == resid_season) & f["fp_r3"].notna()].copy()
    model = lgb.LGBMRegressor(**_PARAMS).fit(train[cols], train[stat])
    test["pred"] = model.predict(test[cols])
    test["residual"] = test[stat] - test["pred"]
    return test


def _project_week(feats: pd.DataFrame, stat: str, positions, season: int, week: int) -> pd.DataFrame:
    """Train on everything before (season, week), project that week's stat."""
    f = _rows(feats, positions)
    cols = feature_columns()
    k = season * 100 + week
    train = f[f["_k"] < k]
    test = f[(f["_k"] == k) & f["fp_r3"].notna()].copy()
    test["pred"] = lgb.LGBMRegressor(**_PARAMS).fit(train[cols], train[stat]).predict(test[cols])
    return test[["player_display_name", "position", "recent_team", "pred"]]


def price_props(prop_df: pd.DataFrame, season: int, week: int,
                feats: pd.DataFrame | None = None, threshold: float = 0.0) -> pd.DataFrame:
    """Price a table of prop lines against model projections.

    prop_df columns: player, market, line, over_odds, under_odds (American odds).
    Returns one row per prop with model P(over), market fair P(over), edge, and
    the best-side EV per 1u; sorted by EV, kept where EV > `threshold`.
    """
    feats = build_features(seasons=list(range(FIRST_SEASON, season + 1))) if feats is None else feats
    out = []
    for market, mdf in prop_df.groupby("market"):
        if market not in MARKETS:
            continue
        positions = MARKETS[market]
        resid = _oos_residuals(feats, market, positions, season - 1)["residual"].to_numpy()
        proj = _project_week(feats, market, positions, season, week)
        proj_idx = {_norm(n): p for n, p in zip(proj["player_display_name"], proj["pred"])}
        for _, r in mdf.iterrows():
            pred = proj_idx.get(_norm(r["player"]))
            if pred is None:
                continue
            p_over = float(_prob_over(resid, np.array([pred]), np.array([r["line"]]))[0])
            fair_over = american_to_prob([r["over_odds"]])[0]
            fair = fair_over / (fair_over + american_to_prob([r["under_odds"]])[0])
            side, ev = _ev_side(p_over, float(r["over_odds"]), float(r["under_odds"]))
            out.append({
                "player": r["player"], "market": market, "line": r["line"],
                "proj": round(pred, 1), "model_p_over": round(p_over, 3),
                "market_p_over": round(float(fair), 3), "edge": round(p_over - float(fair), 3),
                "bet": side, "odds": int(r[f"{side}_odds"]), "ev_per_1u": round(ev, 3),
            })
    res = pd.DataFrame(out)
    return res[res["ev_per_1u"] > threshold].sort_values("ev_per_1u", ascending=False).reset_index(drop=True)


def calibrate(feats: pd.DataFrame, market: str, pool_season: int, eval_season: int) -> dict:
    """Honest interval-coverage of a market's P(over): residual pool from one
    season, coverage checked on a *different, later* season -- exactly what
    price_props does (pool from the prior season, applied to the target). If the
    intervals cover at their nominal rate, the P(over) driving every edge holds.
    """
    positions = MARKETS[market]
    resid = _oos_residuals(feats, market, positions, pool_season)["residual"].to_numpy()
    ev = _oos_residuals(feats, market, positions, eval_season)
    pred, actual = ev["pred"].to_numpy(), ev[market].to_numpy()
    lo, hi = pred + np.percentile(resid, 10), pred + np.percentile(resid, 90)
    q25, q75 = pred + np.percentile(resid, 25), pred + np.percentile(resid, 75)
    pit = np.array([(resid < a - p).mean() for p, a in zip(pred, actual)])
    return {"market": market, "n": len(ev),
            "cover80": round(float(((actual >= lo) & (actual <= hi)).mean()), 3),
            "cover50": round(float(((actual >= q25) & (actual <= q75)).mean()), 3),
            "mean_pit": round(float(pit.mean()), 3)}


def main() -> None:
    import argparse
    from .ingest import current_nfl_season

    p = argparse.ArgumentParser(prog="python -m ffdata.props",
                                description="Player-props edge finder (you supply the lines).")
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--season", type=int, default=current_nfl_season())
    p.add_argument("--props", required=True,
                   help="CSV: player,market,line,over_odds,under_odds")
    p.add_argument("--min-ev", type=float, default=0.0, help="minimum EV per 1u to show")
    args = p.parse_args()

    prop_df = pd.read_csv(args.props)
    print(f"Pricing {len(prop_df)} props for {args.season} week {args.week}...", flush=True)
    edges = price_props(prop_df, args.season, args.week, threshold=args.min_ev)
    if edges.empty:
        print("No +EV props found.")
        return
    print(f"\n+EV bets (of {len(prop_df)} priced):\n")
    print(edges.to_string(index=False))


if __name__ == "__main__":
    main()
