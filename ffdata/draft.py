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
from .scoring import HALF_PPR, PPR, STANDARD, ScoringRules, score

POSITIONS = ("QB", "RB", "WR", "TE")
# Named scoring presets for the CLIs. Any ScoringRules works via the API.
_RULES = {"ppr": PPR, "half": HALF_PPR, "standard": STANDARD}
DEFAULT_LEAGUE = {"teams": 12, "budget": 200, "roster_spots": 15,
                  "starters": {"QB": 1, "RB": 2, "WR": 3, "TE": 1}, "flex": 1}

# Prior-season aggregates + preseason context used to predict next-season points.
_FEATS = ["p_games", "p_fp", "p_ppg", "p_targets", "p_carries", "p_receptions",
          "p_rec_yds", "p_rush_yds", "p_pass_yds", "p_pass_tds", "p_rush_tds",
          "p_rec_tds", "p_tgt_share", "age", "years_exp",
          "team_changed", "coach_changed", "sos"] + [f"is_{p}" for p in POSITIONS]
_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
               subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=4)
# The GBM alone ranks slightly *worse* than raw prior-season points (it chases
# breakouts); a blend beats both -- the model handles age/injury/regression, the
# prior-year anchor keeps proven volume honest. Validated on 2023-24.
_BLEND = 0.4  # weight on the model; 1 - _BLEND on prior-season total


def _season_agg(con, rules: ScoringRules = PPR) -> pd.DataFrame:
    """Per player-season regular-season totals, scored under `rules`.

    Fantasy points come from scoring.score() over the raw weekly stats -- the
    same league-agnostic path the weekly tools use -- not the precomputed
    `fantasy_points_ppr` column, so draft/dynasty values honor any ScoringRules.
    With the default (PPR) this reproduces the old `fantasy_points_ppr` totals.
    """
    weekly = con.sql("""
        select * from weekly
        where season_type = 'REG' and position in ('QB','RB','WR','TE')
    """).df()
    weekly = score(weekly, rules, col="fp")
    return (
        weekly.groupby(["player_id", "season"], as_index=False)
        .agg(position=("position", "first"), player=("player_display_name", "first"),
             games=("fp", "size"), fp=("fp", "sum"), targets=("targets", "sum"),
             carries=("carries", "sum"), receptions=("receptions", "sum"),
             rec_yds=("receiving_yards", "sum"), rush_yds=("rushing_yards", "sum"),
             pass_yds=("passing_yards", "sum"), pass_tds=("passing_tds", "sum"),
             rush_tds=("rushing_tds", "sum"), rec_tds=("receiving_tds", "sum"),
             tgt_share=("target_share", "mean"))
    )


def _roster_info(con) -> pd.DataFrame:
    return con.sql("""
        select gsis_id as player_id, season, max(years_exp) as years_exp,
               max(extract(year from birth_date)) as birth_year
        from rosters group by gsis_id, season
    """).df()


def _team_season(con) -> pd.DataFrame:
    """The team a player is rostered on, per season (for roster-change signal).

    Sourced from `rosters`, not game logs, so it's available in the preseason
    (e.g. a 2026 draft, before any 2026 games have been played)."""
    return con.sql("""
        select player_id, season, team from (
            select gsis_id as player_id, season, team, count(*) c,
                   row_number() over (partition by gsis_id, season order by count(*) desc) rn
            from rosters where team is not null and gsis_id is not null
            group by gsis_id, season, team)
        where rn = 1
    """).df()


def _team_coach(con) -> pd.DataFrame:
    """Head coach per team per season, derived from the schedule (for coach-change)."""
    return con.sql("""
        select season, home_team as team, any_value(home_coach) as coach
        from schedules where game_type = 'REG' group by season, home_team
    """).df()


def _sos(con, rules: ScoringRules = PPR) -> pd.DataFrame:
    """Strength of schedule: for each team-season-position, the average fantasy
    points its upcoming opponents allowed to that position the *prior* year.
    Higher = easier schedule. Uses the known schedule + last year's defenses,
    so it's available at draft time and leak-free. Points-allowed is scored under
    the same `rules` as everything else for consistency."""
    weekly = con.sql("""
        select * from weekly
        where season_type='REG' and position in ('QB','RB','WR','TE')
    """).df()
    weekly = score(weekly, rules, col="fp")
    allowed = (
        weekly.groupby(["opponent_team", "position", "season"], as_index=False)["fp"]
        .sum().rename(columns={"opponent_team": "opp", "fp": "allowed"})
    )
    opp = con.sql("""
        select season, home_team as team, away_team as opp from schedules where game_type='REG'
        union all
        select season, away_team as team, home_team as opp from schedules where game_type='REG'
    """).df()
    prev = allowed.assign(season=allowed["season"] + 1)  # last year's D -> this year's SOS
    m = opp.merge(prev, on=["season", "opp"])
    sos = m.groupby(["team", "season", "position"])["allowed"].mean().reset_index()
    return sos.rename(columns={"allowed": "sos"})


def _feature_frame(con, rules: ScoringRules = PPR) -> pd.DataFrame:
    """Feature rows: prior-season aggregates (S) + preseason context at S+1 ->
    target = fp at S+1. Target is NaN for the not-yet-played season."""
    agg = _season_agg(con, rules)
    ts, coach, sos = _team_season(con), _team_coach(con), _sos(con, rules)
    feat = agg.rename(columns={
        "games": "p_games", "fp": "p_fp", "targets": "p_targets", "carries": "p_carries",
        "receptions": "p_receptions", "rec_yds": "p_rec_yds", "rush_yds": "p_rush_yds",
        "pass_yds": "p_pass_yds", "pass_tds": "p_pass_tds", "rush_tds": "p_rush_tds",
        "rec_tds": "p_rec_tds", "tgt_share": "p_tgt_share"}).copy()
    feat["p_ppg"] = feat["p_fp"] / feat["p_games"].clip(lower=1)
    feat["tseason"] = feat["season"] + 1

    tgt = agg[["player_id", "season", "fp"]].rename(columns={"season": "tseason", "fp": "target_fp"})
    df = feat.merge(tgt, on=["player_id", "tseason"], how="left")
    df = df.merge(_roster_info(con).rename(columns={"season": "tseason"}), on=["player_id", "tseason"], how="left")
    df["age"] = df["tseason"] - df["birth_year"]

    # Roster change: player's team at S+1 differs from S.
    df = df.merge(ts.rename(columns={"team": "prior_team"}), on=["player_id", "season"], how="left")
    df = df.merge(ts.rename(columns={"season": "tseason", "team": "new_team"}), on=["player_id", "tseason"], how="left")
    df["team_changed"] = (df["prior_team"].fillna("") != df["new_team"].fillna("")).astype(int)

    # Coaching change: new team's coach at S+1 differs from that team's coach at S.
    cn = coach.rename(columns={"season": "tseason", "team": "new_team", "coach": "coach_new"})
    co = coach.assign(tseason=coach["season"] + 1).rename(columns={"team": "new_team", "coach": "coach_old"})
    df = df.merge(cn, on=["tseason", "new_team"], how="left")
    df = df.merge(co[["new_team", "tseason", "coach_old"]], on=["new_team", "tseason"], how="left")
    df["coach_changed"] = ((df["coach_new"] != df["coach_old"]) & df["coach_old"].notna()).astype(int)

    # Strength of schedule for the new team + position at S+1.
    df = df.merge(sos.rename(columns={"season": "tseason", "team": "new_team"}),
                  on=["tseason", "new_team", "position"], how="left")
    df["sos"] = df["sos"].fillna(df["sos"].median())

    for p in POSITIONS:
        df[f"is_{p}"] = (df["position"] == p).astype(int)
    return df


def project_season(target_season: int, rules: ScoringRules = PPR, con=None) -> pd.DataFrame:
    """Project every returning player's total points for `target_season`.

    Trains on pairs whose target season is strictly before `target_season`
    (leak-free), then predicts the players entering `target_season`. `rules`
    sets the scoring the projection is expressed in (default PPR).
    """
    con = con or connect()
    df = _feature_frame(con, rules)
    train = df[(df["tseason"] < target_season) & df["target_fp"].notna()]
    test = df[df["tseason"] == target_season].copy()
    if test.empty:
        return test
    model = lgb.LGBMRegressor(**_PARAMS).fit(train[_FEATS], train["target_fp"])
    model_pts = np.clip(model.predict(test[_FEATS]), 0, None)
    # Blend with prior-season total (both are season-point scale).
    test["proj"] = (_BLEND * model_pts + (1 - _BLEND) * test["p_fp"]).clip(lower=0).round(1)
    return test[["player_id", "player", "position", "proj"]].sort_values("proj", ascending=False)


# --------------------------------------------------------------------------- #
# Rookies: draft-capital model (returning-player model can't touch them --
# they have no prior season). SCAFFOLDED but not yet backtested on real data;
# validate with backtest_rookies() before trusting the magnitudes.
# --------------------------------------------------------------------------- #
_ROOKIE_FEATS = ["pick", "log_pick", "draft_round"] + [f"is_{p}" for p in POSITIONS]
_ROOKIE_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=16,
                      min_child_samples=15, subsample=0.9, colsample_bytree=0.9,
                      random_state=0, verbose=-1, n_jobs=4)


def _has_view(con, name: str) -> bool:
    return name in {r[0] for r in con.sql("show tables").fetchall()}


def _draft_capital(con) -> pd.DataFrame | None:
    """Per-player draft capital: season drafted, round, overall pick, position.

    Returns None if the `draft_picks` source hasn't been ingested, so callers can
    degrade gracefully (rookies simply won't appear on the board).
    """
    if not _has_view(con, "draft_picks"):
        return None
    df = con.sql("select * from draft_picks").df()
    if df.empty or "gsis_id" not in df.columns or "season" not in df.columns:
        return None
    # nflverse has renamed the name column across versions; take what's present.
    name_col = next((c for c in ("pfr_player_name", "full_name", "player_name", "player")
                     if c in df.columns), None)
    out = pd.DataFrame({
        "player_id": df["gsis_id"],
        "draft_season": pd.to_numeric(df["season"], errors="coerce"),
        "draft_round": pd.to_numeric(df.get("round"), errors="coerce"),
        "pick": pd.to_numeric(df.get("pick"), errors="coerce"),
        "position": df.get("position"),
        "player": df[name_col] if name_col else df["gsis_id"],
    }).dropna(subset=["player_id", "draft_season", "pick"])
    return out[out["position"].isin(POSITIONS)].reset_index(drop=True)


def _rookie_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pick"] = df["pick"].astype(float)
    df["log_pick"] = np.log(df["pick"].clip(lower=1))
    # Fall back to a pick-derived round if the source didn't carry one.
    df["draft_round"] = df["draft_round"].fillna(df["pick"] // 32 + 1).astype(float)
    for p in POSITIONS:
        df[f"is_{p}"] = (df["position"] == p).astype(int)
    return df


def rookie_projection(target_season: int, rules: ScoringRules = PPR, con=None) -> pd.DataFrame | None:
    """Project incoming rookies' rookie-season points from draft capital.

    Trains draft-capital -> rookie-season points on all rookies drafted BEFORE
    `target_season` (leak-free: draft position is preseason-known), then predicts
    the players drafted INTO `target_season`. Returns None if the `draft_picks`
    source isn't ingested; an empty frame if there are no rookies to project.

    NOTE: scaffolded, not yet validated on real data -- run backtest_rookies().
    """
    con = con or connect()
    caps = _draft_capital(con)
    if caps is None:
        return None
    empty = pd.DataFrame(columns=["player_id", "player", "position", "proj"])
    fp = _season_agg(con, rules)[["player_id", "season", "fp"]]
    # A rookie's rookie-season points = production in the season they were drafted.
    train = caps.merge(fp, left_on=["player_id", "draft_season"],
                       right_on=["player_id", "season"], how="inner")
    train = _rookie_features(train[train["draft_season"] < target_season])
    test = _rookie_features(caps[caps["draft_season"] == target_season].copy())
    if train.empty or test.empty:
        return empty
    model = lgb.LGBMRegressor(**_ROOKIE_PARAMS).fit(train[_ROOKIE_FEATS], train["fp"])
    test["proj"] = np.clip(model.predict(test[_ROOKIE_FEATS]), 0, None).round(1)
    return test[["player_id", "player", "position", "proj"]].sort_values("proj", ascending=False)


def backtest_rookies(target_season: int, rules: ScoringRules = PPR, con=None) -> dict | None:
    """Rank quality of rookie projections vs their actual rookie-season finish.

    Returns None if `draft_picks` isn't ingested. This is the measurement the
    rookie model still needs -- run it on a real lake before trusting the board's
    rookie values.
    """
    from scipy.stats import spearmanr
    con = con or connect()
    proj = rookie_projection(target_season, rules=rules, con=con)
    if proj is None:
        return None
    actual = _season_agg(con, rules)
    actual = actual[actual["season"] == target_season][["player_id", "fp"]]
    m = proj.merge(actual, on="player_id", how="inner")
    if len(m) < 3:
        return {"season": target_season, "n": len(m), "note": "too few rookies with data"}
    return {"season": target_season, "n": len(m),
            "rookie_spearman": round(float(spearmanr(m["proj"], m["fp"]).correlation), 3),
            "rookie_mae": round(float((m["proj"] - m["fp"]).abs().mean()), 1)}


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


def draft_board(target_season: int, league: dict | None = None,
                rules: ScoringRules = PPR, include_rookies: bool = True,
                con=None) -> pd.DataFrame:
    """Ranked draft board: season projection, VOR, and auction dollar value.

    `rules` sets the league scoring (default PPR); VOR and auction $ follow from
    the scored projections, so the whole board reflects the chosen scoring.
    `include_rookies` folds in the draft-capital rookie model when the
    `draft_picks` source is available (silently veterans-only if it isn't).
    """
    league = league or DEFAULT_LEAGUE
    con = con or connect()
    proj = project_season(target_season, rules=rules, con=con)
    if include_rookies:
        rookies = rookie_projection(target_season, rules=rules, con=con)
        if rookies is not None and not rookies.empty:
            proj = (pd.concat([proj, rookies], ignore_index=True)
                    .drop_duplicates(subset="player_id", keep="first"))
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


def round_cost(board: pd.DataFrame, rnd: int, teams: int = 12) -> float:
    """Auction $ a round-`rnd` pick is worth (value at the top of that round)."""
    rank = max(0, (rnd - 1) * teams)
    return float(board.iloc[rank]["auction"]) if rank < len(board) else 1.0


def keeper_value(board: pd.DataFrame, keepers: list, teams: int = 12,
                 cost_type: str = "auction") -> pd.DataFrame:
    """Surplus value of keeping each player at its cost.

    keepers: list of (player, cost). cost is auction $ (cost_type="auction") or a
    draft round (cost_type="round", converted via round_cost). Surplus = the
    player's projected value minus what keeping him costs; positive is a bargain.
    """
    from .optimize import _norm
    bi = {_norm(r["player"]): r for _, r in board.iterrows()}
    rows = []
    for name, cost in keepers:
        r = bi.get(_norm(name))
        if r is None:
            continue
        c = float(cost) if cost_type == "auction" else round_cost(board, int(cost), teams)
        rows.append({"player": r["player"], "position": r["position"],
                     "value": int(r["auction"]), "cost": int(round(c)),
                     "surplus": int(round(r["auction"] - c))})
    return pd.DataFrame(rows).sort_values("surplus", ascending=False).reset_index(drop=True)


def trade_value(board: pd.DataFrame, side_a: list, side_b: list) -> dict:
    """Evaluate a trade by total projected value (auction $ and VOR) per side."""
    from .optimize import _norm
    bi = {_norm(r["player"]): r for _, r in board.iterrows()}

    def side(names):
        got = [bi[_norm(n)] for n in names if _norm(n) in bi]
        return {"players": [{"player": r["player"], "position": r["position"],
                             "value": int(r["auction"]), "vor": float(r["vor"])} for r in got],
                "auction": sum(int(r["auction"]) for r in got),
                "vor": round(sum(float(r["vor"]) for r in got), 1)}

    a, b = side(side_a), side(side_b)
    diff = a["auction"] - b["auction"]
    verdict = ("roughly even" if abs(diff) <= 5 else
               f"Side A wins by ${diff}" if diff > 0 else f"Side B wins by ${-diff}")
    return {"side_a": a, "side_b": b, "diff": diff, "verdict": verdict}


def backtest_rank(target_season: int, rules: ScoringRules = PPR, con=None) -> dict:
    """Rank quality of the preseason projection vs the actual season finish."""
    from scipy.stats import spearmanr
    con = con or connect()
    proj = project_season(target_season, rules=rules, con=con)
    actual = _season_agg(con, rules)
    actual = actual[actual["season"] == target_season][["player_id", "fp"]]
    m = proj.merge(actual, on="player_id", how="inner")
    naive = _feature_frame(con, rules)
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
    p.add_argument("--scoring", choices=list(_RULES), default="ppr")
    p.add_argument("--position", choices=list(POSITIONS))
    p.add_argument("--drafted", default="", help="comma-separated already-drafted players")
    p.add_argument("--no-rookies", action="store_true",
                   help="exclude the draft-capital rookie model (needs the draft_picks source)")
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()
    board = draft_board(args.season, rules=_RULES[args.scoring], include_rookies=not args.no_rookies)
    if board.empty:
        raise SystemExit(f"No draftable data for {args.season}.")
    avail = best_available(board, args.drafted.split(",") if args.drafted else [], args.position, args.n)
    pd.set_option("display.width", 100)
    print(f"\nDraft board {args.season} ({args.scoring.upper()}; "
          f"VOR = value over replacement, $ = auction value):\n")
    print(avail[["player", "position", "proj", "vor", "auction"]].to_string(index=False))
