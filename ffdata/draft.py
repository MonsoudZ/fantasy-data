"""Draft engine: preseason season-long value, VOR rankings, snake + auction.

Drafting is a different problem from the weekly optimizer. A draft happens in the
preseason -- no current-season data exists -- and you care about a player's whole
*season*, not one week. So this is a separate model: predict a player's full-
season fantasy points from PRIOR-season production + age + experience, trained on
consecutive-season pairs (2019->2020, ..., 2023->2024).

From those season projections everything a draft needs follows:

  * VOR (value over replacement): a player's projected points minus the
    replacement-level player at his position, given league size and starters --
    the right cross-position currency (a 250-pt RB and a 250-pt QB aren't equal
    because QBs are deeper).
  * snake: the best available player by VOR given who's already gone.
  * auction: VOR converted to dollar values for a budget.

Keepers / trades / dynasty are the same value engine applied differently and are
natural extensions. Cold-start limit: rookies have no prior season, so this model
skips them (they need a draft-capital model).

    from ffdata.draft import draft_board, best_available
    board = draft_board(2024)                    # ranked, with VOR + auction $
    print(best_available(board, drafted=["Ja'Marr Chase"]).head(10))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from .db import connect

POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_LEAGUE = {"teams": 12, "budget": 200, "roster_spots": 15,
                  "starters": {"QB": 1, "RB": 2, "WR": 3, "TE": 1}, "flex": 1}

# Prior-season aggregates used to predict next-season points.
_FEATS = ["p_games", "p_fp", "p_ppg", "p_targets", "p_carries", "p_receptions",
          "p_rec_yds", "p_rush_yds", "p_pass_yds", "p_pass_tds", "p_rush_tds",
          "p_rec_tds", "p_tgt_share", "age", "years_exp"] + [f"is_{p}" for p in POSITIONS]
_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
               subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=4)
# The GBM alone ranks slightly *worse* than raw prior-season points (it chases
# breakouts); a blend beats both -- the model handles age/injury/regression, the
# prior-year anchor keeps proven volume honest. Validated on 2023-24.
_BLEND = 0.4  # weight on the model; 1 - _BLEND on prior-season total


def _season_agg(con) -> pd.DataFrame:
    """Per player-season regular-season totals (PPR)."""
    return con.sql("""
        select player_id, season, any_value(position) as position,
               any_value(player_display_name) as player, count(*) as games,
               sum(fantasy_points_ppr) as fp, sum(targets) as targets,
               sum(carries) as carries, sum(receptions) as receptions,
               sum(receiving_yards) as rec_yds, sum(rushing_yards) as rush_yds,
               sum(passing_yards) as pass_yds, sum(passing_tds) as pass_tds,
               sum(rushing_tds) as rush_tds, sum(receiving_tds) as rec_tds,
               avg(target_share) as tgt_share
        from weekly where season_type = 'REG' and position in ('QB','RB','WR','TE')
        group by player_id, season
    """).df()


def _roster_info(con) -> pd.DataFrame:
    return con.sql("""
        select gsis_id as player_id, season, max(years_exp) as years_exp,
               max(extract(year from birth_date)) as birth_year
        from rosters group by gsis_id, season
    """).df()


def _pairs(agg: pd.DataFrame, ri: pd.DataFrame) -> pd.DataFrame:
    """Feature rows: prior-season aggregates (season S) -> target = fp at S+1."""
    feat = agg.rename(columns={
        "games": "p_games", "fp": "p_fp", "targets": "p_targets", "carries": "p_carries",
        "receptions": "p_receptions", "rec_yds": "p_rec_yds", "rush_yds": "p_rush_yds",
        "pass_yds": "p_pass_yds", "pass_tds": "p_pass_tds", "rush_tds": "p_rush_tds",
        "rec_tds": "p_rec_tds", "tgt_share": "p_tgt_share"}).copy()
    feat["p_ppg"] = feat["p_fp"] / feat["p_games"].clip(lower=1)
    feat["tseason"] = feat["season"] + 1
    tgt = agg[["player_id", "season", "fp"]].rename(columns={"season": "tseason", "fp": "target_fp"})
    df = feat.merge(tgt, on=["player_id", "tseason"], how="left")  # target None until fit time
    # age / experience as of the *target* season (known preseason)
    df = df.merge(ri.rename(columns={"season": "tseason"}), on=["player_id", "tseason"], how="left")
    df["age"] = df["tseason"] - df["birth_year"]
    for p in POSITIONS:
        df[f"is_{p}"] = (df["position"] == p).astype(int)
    return df


def project_season(target_season: int, con=None) -> pd.DataFrame:
    """Project every returning player's total points for `target_season`.

    Trains on pairs whose target season is strictly before `target_season`
    (leak-free), then predicts the players entering `target_season`.
    """
    con = con or connect()
    agg, ri = _season_agg(con), _roster_info(con)
    df = _pairs(agg, ri)
    train = df[(df["tseason"] < target_season) & df["target_fp"].notna()]
    test = df[df["tseason"] == target_season].copy()
    if test.empty:
        return test
    model = lgb.LGBMRegressor(**_PARAMS).fit(train[_FEATS], train["target_fp"])
    model_pts = np.clip(model.predict(test[_FEATS]), 0, None)
    # Blend with prior-season total (both are season-point scale).
    test["proj"] = (_BLEND * model_pts + (1 - _BLEND) * test["p_fp"]).clip(lower=0).round(1)
    return test[["player_id", "player", "position", "proj"]].sort_values("proj", ascending=False)


def _replacement_ranks(league: dict) -> dict:
    """The rank at each position below which a player is 'replacement level'."""
    t, s = league["teams"], league["starters"]
    base = {p: t * s.get(p, 0) for p in POSITIONS}
    # spread FLEX slots across RB/WR/TE by their share of starting demand
    flex_pool = t * league.get("flex", 0)
    fx = ["RB", "WR", "TE"]
    denom = sum(base[p] for p in fx) or 1
    for p in fx:
        base[p] += round(flex_pool * base[p] / denom)
    return base


def draft_board(target_season: int, league: dict | None = None, con=None) -> pd.DataFrame:
    """Ranked draft board: season projection, VOR, and auction dollar value."""
    league = league or DEFAULT_LEAGUE
    proj = project_season(target_season, con=con)
    if proj.empty:
        return proj
    repl_rank = _replacement_ranks(league)
    proj = proj.copy()
    repl_pts = {}
    for p in POSITIONS:
        pos = proj[proj["position"] == p].sort_values("proj", ascending=False).reset_index(drop=True)
        r = min(repl_rank[p], len(pos) - 1)
        repl_pts[p] = float(pos.loc[r, "proj"]) if len(pos) else 0.0
    proj["vor"] = (proj["proj"] - proj["position"].map(repl_pts)).round(1)

    # Auction: distribute the budget (above a $1 minimum per roster spot) by positive VOR.
    pool = league["teams"] * (league["budget"] - league["roster_spots"])
    pos_vor = proj["vor"].clip(lower=0)
    total = pos_vor.sum() or 1
    proj["auction"] = (1 + pos_vor / total * pool).round(0).astype(int)
    return proj.sort_values("vor", ascending=False).reset_index(drop=True)


def best_available(board: pd.DataFrame, drafted: list[str] | None = None, position: str | None = None,
                   n: int = 15) -> pd.DataFrame:
    """Top remaining players by VOR, excluding already-drafted names."""
    from .optimize import _norm
    taken = {_norm(x) for x in (drafted or [])}
    out = board[~board["player"].map(lambda s: _norm(s) in taken)]
    if position:
        out = out[out["position"] == position]
    return out.head(n).reset_index(drop=True)


def backtest_rank(target_season: int, con=None) -> dict:
    """Rank quality of the preseason projection vs the actual season finish."""
    from scipy.stats import spearmanr
    con = con or connect()
    proj = project_season(target_season, con=con)
    actual = _season_agg(con)
    actual = actual[actual["season"] == target_season][["player_id", "fp"]]
    m = proj.merge(actual, on="player_id", how="inner")
    naive = _pairs(_season_agg(con), _roster_info(con))
    naive = naive[naive["tseason"] == target_season][["player_id", "p_fp"]]
    m = m.merge(naive, on="player_id", how="left")
    return {"season": target_season, "n": len(m),
            "model_spearman": round(float(spearmanr(m["proj"], m["fp"]).correlation), 3),
            "lastyear_spearman": round(float(spearmanr(m["p_fp"], m["fp"]).correlation), 3),
            "model_mae": round(float((m["proj"] - m["fp"]).abs().mean()), 1)}


if __name__ == "__main__":
    import argparse
    from .ingest import current_nfl_season
    p = argparse.ArgumentParser(prog="python -m ffdata.draft", description="Draft board / value rankings")
    p.add_argument("--season", type=int, default=current_nfl_season())
    p.add_argument("--position", choices=list(POSITIONS))
    p.add_argument("--drafted", default="", help="comma-separated already-drafted players")
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()
    board = draft_board(args.season)
    if board.empty:
        raise SystemExit(f"No draftable data for {args.season}.")
    avail = best_available(board, args.drafted.split(",") if args.drafted else [], args.position, args.n)
    pd.set_option("display.width", 100)
    print(f"\nDraft board {args.season} (VOR = value over replacement, $ = auction value):\n")
    print(avail[["player", "position", "proj", "vor", "auction"]].to_string(index=False))
