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

import logging

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

from .db import connect
from .gbm import gbm_params
from .scoring import HALF_PPR, PPR, STANDARD, ScoringRules, score
from .sleeper import LIVE_SEVERE, norm_name

_log = logging.getLogger("ffdata.draft")

POSITIONS = ("QB", "RB", "WR", "TE")
# draft_picks ships PFR team codes; everything else in the lake uses nflverse's.
_PFR_TEAM = {"GNB": "GB", "KAN": "KC", "LAR": "LA", "LVR": "LV",
             "NOR": "NO", "NWE": "NE", "SFO": "SF", "TAM": "TB"}
# Named scoring presets for the CLIs. Any ScoringRules works via the API.
_RULES = {"ppr": PPR, "half": HALF_PPR, "standard": STANDARD}
DEFAULT_LEAGUE = {"teams": 12, "budget": 200, "roster_spots": 15,
                  "starters": {"QB": 1, "RB": 2, "WR": 3, "TE": 1}, "flex": 1}

# Prior-season aggregates + preseason context used to predict next-season points.
_FEATS = ["p_games", "p_fp", "p_ppg", "p_targets", "p_carries", "p_receptions",
          "p_rec_yds", "p_rush_yds", "p_pass_yds", "p_pass_tds", "p_rush_tds",
          "p_rec_tds", "p_tgt_share", "age", "years_exp",
          "team_changed", "coach_changed", "sos"] + [f"is_{p}" for p in POSITIONS]
_PARAMS = gbm_params(n_estimators=400, num_leaves=31, min_child_samples=20)
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
                   -- `team` breaks the tie: a player with equal week counts on two
                   -- teams would otherwise get an arbitrary one, and DuckDB is
                   -- multi-threaded, so "arbitrary" differs between runs. That
                   -- flips team_changed/sos and makes the whole board irreproducible.
                   row_number() over (partition by gsis_id, season
                                      order by count(*) desc, team) rn
            from rosters where team is not null and gsis_id is not null
            group by gsis_id, season, team)
        where rn = 1
    """).df()


def _team_coach(con) -> pd.DataFrame:
    """Head coach per team per season, derived from the schedule (for coach-change).

    The coach of the team's LAST regular-season game -- not `min()`/`any_value()`.
    A mid-season firing leaves a team with two coaches for the year; who they
    ENDED with is the one they carry into the offseason, so it's the right anchor
    for `new_coach`. Ordering by week desc is also deterministic (one home game
    per week), which `any_value()` was not (it flipped run to run and made the
    board irreproducible)."""
    return con.sql("""
        select season, team, coach from (
            select season, home_team as team, home_coach as coach,
                   row_number() over (partition by season, home_team order by week desc) rn
            from schedules where game_type = 'REG' and home_coach is not null)
        where rn = 1
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
# Rookies: draft-capital model (the returning-player model can't touch them --
# they have no prior season).
#
# BACKTESTED (2022-25, ~70 rookies/yr). Where a player was drafted is nearly the
# whole signal: sorting by pick alone ranks at 0.575 Spearman, and the original
# multi-feature GBM (pick + round + position, 300 trees) managed only 0.510 --
# it overfit ~350 training rows and wiggled non-monotonically in pick. What ships
# is a small pick-only curve with a monotone constraint (earlier pick can never
# project lower): 0.566, i.e. it matches the naive ordering while still emitting
# POINTS, which VOR and auction $ need. Position is deliberately excluded -- both
# as GBM features (0.510) and as a post-hoc per-position scale (0.520) it made
# ranking measurably worse on this sample. Rookie seasons are genuinely noisy:
# expect ~0.57 rank and ~45 pts MAE, so treat rookie values as a prior, not a
# projection. Re-check with backtest_rookies().
# --------------------------------------------------------------------------- #
_ROOKIE_FEATS = ["pick", "log_pick"]
# Capacity above this changes nothing (rank is flat at 0.566 from 4 to 31 leaves);
# the curve is genuinely step-shaped because ~350 rows only support a few honest
# splits on pick. Ties are therefore real -- broken by pick, the better signal.
_ROOKIE_PARAMS = gbm_params(n_estimators=150, num_leaves=8, min_child_samples=15,
                            subsample=0.9, colsample_bytree=0.9,
                            monotone_constraints=[-1, -1])  # later pick -> never higher


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
    # draft_picks uses PFR team codes (GNB/KAN/LVR); the rest of the lake uses
    # nflverse ones (GB/KC/LV). Unmapped, 8 teams silently lose context -- and if
    # the column is missing entirely, EVERY rookie's team goes blank and the
    # situation join matches nothing, so say so rather than failing silently.
    if "team" in df.columns:
        team = df["team"].map(lambda t: _PFR_TEAM.get(t, t))
    else:
        _log.warning("draft_picks has no 'team' column; rookie situation context "
                     "(vacated/returning/blocked_by) will be blank")
        team = pd.Series(index=df.index, dtype=object)
    out = pd.DataFrame({
        "player_id": df["gsis_id"],
        "draft_season": pd.to_numeric(df["season"], errors="coerce"),
        "draft_round": pd.to_numeric(df.get("round"), errors="coerce"),
        "pick": pd.to_numeric(df.get("pick"), errors="coerce"),
        "position": df.get("position"),
        "player": df[name_col] if name_col else df["gsis_id"],
        "team": team,
    }).dropna(subset=["player_id", "draft_season", "pick"])
    return out[out["position"].isin(POSITIONS)].reset_index(drop=True)


def rookie_context(target_season: int, rules: ScoringRules = PPR, con=None) -> pd.DataFrame | None:
    """Opportunity context for each incoming rookie -- what he's walking into.

    For the drafting team and position: how much of last season's production
    LEFT (free agency/trade/retirement) vs stayed (the competition he must beat),
    plus his spot on the preseason depth chart.

    This is deliberately NOT fed to the projection. Measured on 2022-25, these
    signals are real but weak (vacated production correlates +0.14 with rookie
    points, returning competition -0.09) next to draft pick (+0.62) -- and teams
    already draft partly for need (QB +0.31, TE +0.23), so pick absorbs some of
    it. Every variant that modeled them ranked WORSE than pick alone (0.54 vs
    0.57) on ~350 training rows. So it ships as context for a human to weigh,
    not as a number the model pretends to know.
    """
    con = con or connect()
    caps = _draft_capital(con)
    if caps is None:
        return None
    rooks = caps[caps["draft_season"] == target_season].copy()
    if rooks.empty:
        return rooks.assign(vacated_fp=[], returning_fp=[], depth_rank=[])

    agg, ts = _season_agg(con, rules), _team_season(con)
    prior = (agg[agg["season"] == target_season - 1][["player_id", "position", "fp"]]
             .merge(ts[ts["season"] == target_season - 1][["player_id", "team"]], on="player_id"))
    now = ts[ts["season"] == target_season][["player_id", "team"]].rename(columns={"team": "team_now"})
    if now.empty:
        # Target-season rosters aren't ingested, so we can't tell who stayed from
        # who left -- vacated/returning would be fiction (every prior player would
        # count as gone). Report them unknown rather than inflated, keeping the
        # signals that don't need the new roster (depth chart, scheme).
        out = rooks.assign(vacated_fp=np.nan, returning_fp=np.nan,
                           blocked_by=np.nan, blocked_by_fp=np.nan)
        out["depth_rank"] = out["player_id"].map(_depth_rank(con, target_season))
        out = out.merge(_team_pass_rate(con, target_season - 1), on="team", how="left")
        cols = ["player", "position", "team", "pick", "vacated_fp", "returning_fp",
                "blocked_by", "blocked_by_fp", "depth_rank", "pass_rate"]
        return out[cols].sort_values("pick").reset_index(drop=True)
    prior = prior.merge(now, on="player_id", how="left")
    prior["stays"] = prior["team_now"] == prior["team"]
    ctx = (prior.assign(gone=np.where(prior["stays"], 0.0, prior["fp"]),
                        back=np.where(prior["stays"], prior["fp"], 0.0))
           .groupby(["team", "position"], as_index=False)
           .agg(vacated_fp=("gone", "sum"), returning_fp=("back", "sum")))

    # WHO is still standing in front of him. A summed "returning" number can't
    # tell an entrenched star from three replaceable bodies -- and that is the
    # difference between a rookie who starts and one who redshirts.
    staying = prior[prior["stays"]].sort_values("fp", ascending=False)
    blocker = (staying.groupby(["team", "position"], as_index=False)
               .agg(blocked_by_id=("player_id", "first"), blocked_by_fp=("fp", "first")))
    names = agg.sort_values("season").groupby("player_id")["player"].last()

    out = rooks.merge(ctx, on=["team", "position"], how="left") \
               .merge(blocker, on=["team", "position"], how="left")
    out[["vacated_fp", "returning_fp"]] = out[["vacated_fp", "returning_fp"]].fillna(0.0).round(1)
    out["blocked_by"] = out["blocked_by_id"].map(names)
    out["blocked_by_fp"] = out["blocked_by_fp"].fillna(0.0).round(1)
    out["depth_rank"] = out["player_id"].map(_depth_rank(con, target_season))
    # Does this offense even throw? A run-heavy team caps every receiver in the
    # room no matter how much production left.
    out = out.merge(_team_pass_rate(con, target_season - 1), on="team", how="left")
    cols = ["player", "position", "team", "pick", "vacated_fp", "returning_fp",
            "blocked_by", "blocked_by_fp", "depth_rank", "pass_rate"]
    return out[cols].sort_values("pick").reset_index(drop=True)


def _team_pass_rate(con, season: int) -> pd.DataFrame:
    """Team pass share of offensive plays in `season` (pass att / (pass + rush)).

    Scheme sets the ceiling: a rookie receiver on a run-first offense competes
    for a smaller pie than the raw vacated-production number suggests.
    """
    try:
        df = con.sql(f"""
            select recent_team as team,
                   sum(coalesce(attempts, 0)) as pass_att,
                   sum(coalesce(carries, 0)) as rush_att
            from weekly where season = {int(season)} and season_type = 'REG'
            group by recent_team
        """).df()
    except Exception:  # noqa: BLE001 - no lake -> no scheme context
        return pd.DataFrame(columns=["team", "pass_rate"])
    total = df["pass_att"] + df["rush_att"]
    df["pass_rate"] = (df["pass_att"] / total.where(total > 0)).round(3)
    return df[["team", "pass_rate"]]


def _depth_rank(con, season: int) -> dict:
    """player_id -> depth-chart rank for `season` (1 = starter), if available.

    The source changed format: older seasons carry `depth_team`, newer ones are
    dated snapshots with `pos_rank`. Take whichever is present.
    """
    if not _has_view(con, "depth_charts"):
        return {}
    try:
        df = con.sql(f"select * from depth_charts where season = {int(season)}").df()
    except Exception:  # noqa: BLE001 - missing/odd schema -> no depth context
        return {}
    rank_col = next((c for c in ("pos_rank", "depth_team") if c in df.columns), None)
    if df.empty or rank_col is None or "gsis_id" not in df.columns:
        return {}
    d = df[["gsis_id", rank_col]].dropna()
    d[rank_col] = pd.to_numeric(d[rank_col], errors="coerce")
    return d.dropna().groupby("gsis_id")[rank_col].min().to_dict()


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

    Backtested on 2022-25: ~0.57 rank, ~45 pts MAE -- about what sorting by draft
    pick achieves, which is the honest ceiling here (see the block comment above).
    Rookie values are a prior, not a projection.
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
    # The curve is stepped, so rookies tie often; break ties by draft pick -- the
    # strongest signal we have -- rather than leaving the order to sort chance.
    ordered = test.sort_values(["proj", "pick"], ascending=[False, True])
    return ordered[["player_id", "player", "position", "proj"]]


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
    """The rank at each position below which a player is 'replacement level'.

    Dedicated starter slots set the base; flex/superflex slots deepen it:
      * FLEX (RB/WR/TE) is spread across those three by their share of demand.
      * SUPERFLEX (QB-eligible) deepens QB -- a startable QB2 is the scarce asset
        those slots target, so we treat superflex as ~a second QB slot (the
        standard VOR approximation; some SF slots do go to RB/WR in practice).
    This is what makes QB value jump in superflex/2-QB leagues instead of being
    priced like a shallow 1-QB league.
    """
    t, s = league["teams"], league["starters"]
    base = {p: t * s.get(p, 0) for p in POSITIONS}

    # A flex pool deepens its eligible positions in proportion to their share of
    # starting demand -- an RB/WR/TE flex mostly deepens whichever of the three the
    # league already starts most of.
    def spread(pool: int, positions: list[str]) -> None:
        denom = sum(base[p] for p in positions) or 1
        for p in positions:
            base[p] += round(pool * base[p] / denom)

    spread(t * league.get("flex", 0), ["RB", "WR", "TE"])
    spread(t * league.get("wrte", 0), ["WR", "TE"])
    spread(t * league.get("rbwr", 0), ["RB", "WR"])
    # SUPERFLEX (QB-eligible) deepens QB -- a startable QB2 is the scarce asset
    # those slots target, which is what lifts QB value in 2-QB/superflex leagues.
    base["QB"] += t * league.get("superflex", 0)
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
    from .ingest import upcoming_nfl_season
    p = argparse.ArgumentParser(prog="python -m ffdata.draft", description="Draft board / value rankings")
    p.add_argument("--season", type=int, default=upcoming_nfl_season(),
                   help="season to draft for (defaults to the upcoming season)")
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


# Roster codes that mean "not currently available". `status` on the target-season
# roster is the freshest availability signal we have in the offseason -- it
# reflects TODAY, not last December.
#
# CAVEAT on the last four: nflverse populated SUS/RSN/NWT densely in 2019-2020
# (187/177/228 players in 2019 alone) and has carried essentially ZERO since --
# one SUS row in 2022, none in 2021 or 2023-2026. They're kept because they're
# correct where the data exists (2019-20 backtests, and if upstream restores
# them), but a suspension or holdout will NOT surface for a current draft. Don't
# read an empty flag as "nobody is suspended".
_INACTIVE_STATUS = {
    "RET": "retired",
    "RSR": "retired",              # reserve/retired
    "RES": "on injured reserve",
    "PUP": "on PUP",
    "NFI": "non-football injury",
    "E14": "exempt",
    "EXE": "on the exempt list",
    "CUT": "released",
    "UFA": "an unsigned free agent",
    "RFA": "an unsigned restricted FA",
    "SUS": "suspended",            # dormant since 2021 -- see caveat above
    "RSN": "holding out (did not report)",
    "NWT": "holding out (not with team)",
}
# Designations that actually cost a player the game. "Questionable" does not --
# most Questionable players suit up, so treating it as a red flag would light up
# half the league every week.
_MISSED = ("Out", "Doubtful")
# Costs a game but resolves in days, so it says nothing about Week 1 availability
# -- still counted in weeks_out (he did miss it), never flagged as ending hurt.
_TRANSIENT = ("Illness",)


def _live_status(con, season: int) -> pd.DataFrame:
    """Sleeper's live feed joined to our player ids.

    This is the only source we have that knows about a player TODAY -- nflverse's
    injury report ends with last season, and its roster `status` is a coarse
    snapshot. Sleeper carries the current designation plus body part and notes
    ("Surgery"), and it is where suspensions actually live: `injury_status = Sus`.

    Read from the cached `sleeper_status` view, never fetched here -- see
    `sleeper.refresh_live_status`. Absent view (nobody ran the refresh, or no
    network) simply means no live column.

    Joined on name+position, not id: Sleeper populates `gsis_id` for only ~16% of
    rostered players. Both sides go through the SAME `norm_name`, so the keys
    can't drift apart.
    """
    cols = ["player_id", "live_code", "live_body", "live_note", "news_date"]
    try:
        live = con.sql("select * from sleeper_status where live_code is not null").df()
    except duckdb.CatalogException:
        return pd.DataFrame(columns=cols)
    if live.empty:
        return pd.DataFrame(columns=cols)

    ros = con.sql(
        "select distinct gsis_id as player_id, full_name, position from rosters "
        "where season = ? and gsis_id is not null", params=[season]).df()
    ros["name_key"] = ros["full_name"].map(norm_name)
    # A name+position key shared by two players can't be resolved, so it is
    # dropped rather than fanned out onto both of them.
    dupe = ros.duplicated(["name_key", "position"], keep=False)
    ros = ros[~dupe]

    live["gsis_id"] = live["gsis_id"].astype("string").str.strip()
    by_id = live[live["gsis_id"].notna()].merge(
        ros[["player_id"]], left_on="gsis_id", right_on="player_id", how="inner")
    by_name = live.merge(ros[["player_id", "name_key", "position"]],
                         on=["name_key", "position"], how="inner")
    both = pd.concat([by_id, by_name], ignore_index=True)
    return both.drop_duplicates("player_id", keep="first")[cols]


def _team_last_week(con, season: int) -> pd.Series:
    """Each team's final PLAYED week in `season` (regular season + playoffs), from
    the schedule -- the ground truth for how far a team went. Indexed by team."""
    try:
        df = con.sql(
            "select team, max(week) as last_week from ("
            "  select home_team as team, week from schedules "
            "    where season = ? and home_score is not null "
            "  union all "
            "  select away_team as team, week from schedules "
            "    where season = ? and home_score is not null) group by team",
            params=[season, season]).df()
    except Exception:  # noqa: BLE001 - no schedules view -> caller falls back
        return pd.Series(dtype="float64")
    return df.set_index("team")["last_week"]


def availability_context(target_season: int, con=None) -> pd.DataFrame:
    """How last season ENDED for each player, plus how he sits right now.

    A season-total projection quietly assumes a full season. It can't tell you
    that a guy tore something in the divisional round and won't be back until
    November -- the projection just says "he scored 240 last year". This does:

      * weeks_out    -- games he was ruled Out/Doubtful for last season
      * last_injury  -- body part on his most recent Out/Doubtful report
      * last_week    -- and when, with the round (REG/WC/DIV/CON/SB)
      * ended_hurt   -- that report came in his team's final two weeks, i.e. he
                        limped out of the season rather than getting an offseason
      * status       -- current roster status if he isn't ACT (IR, PUP, retired)

    `status` is the one that matters most in July: it's a live snapshot, so a
    player still on IR now is a player whose rehab is running long.

    Context only, like the rest of `player_context` -- the injury report is a
    coach's strategic document as much as a medical one, and modeling it as a
    feature would mostly fit team-level reporting habits.
    """
    con = con or connect()
    prior = target_season - 1
    try:
        inj = con.sql(
            "select gsis_id as player_id, team, week, game_type, report_status, "
            "report_primary_injury from injuries where season = ?",
            params=[prior],
        ).df()
    except duckdb.CatalogException:
        # The dataset simply isn't in the lake: degrade to no notes. Narrower than
        # `except Exception` on purpose -- a renamed upstream column should raise
        # here rather than silently blanking every player's health.
        return pd.DataFrame(columns=["player_id", "weeks_out", "last_injury", "last_week",
                                     "last_round", "ended_hurt", "status",
                                     "live_code", "live_body", "live_note", "news_date"])

    out = pd.DataFrame(columns=["player_id"])
    if not inj.empty:
        inj = inj.dropna(subset=["player_id", "week"])
        # The report doubles as a personal-absence log; those aren't health risks.
        body = inj["report_primary_injury"].fillna("")
        hurt = inj[inj["report_status"].isin(_MISSED)
                   & ~body.str.contains("Not injury related", case=False)]
        # A team's last week depends on how far it went (18 if it missed the
        # playoffs, 22 if it reached the Super Bowl), so "ended the season hurt"
        # only means anything measured against that team's OWN finish -- not the
        # player's last report, which would be trivially true for everyone.
        # Take that finish from the SCHEDULE (ground truth), not from the last
        # week anyone on the team happened to file an injury report -- a deep team
        # with no final-week report would otherwise be given too short a season.
        team_last = _team_last_week(con, prior)
        if team_last.empty:                      # no schedule -> fall back to reports
            team_last = inj.groupby("team")["week"].max()
        season_last = int(team_last.max()) if len(team_last) else inj["week"].max()
        if not hurt.empty:
            last = hurt.sort_values("week").groupby("player_id").tail(1)
            out = last[["player_id", "team", "week", "game_type", "report_primary_injury"]].rename(
                columns={"week": "last_week", "game_type": "last_round",
                         "report_primary_injury": "last_injury"})
            # Distinct weeks: a player can appear on several daily reports per game.
            out["weeks_out"] = out["player_id"].map(
                hurt.groupby("player_id")["week"].nunique())
            # Ruled out in his team's final fortnight: no healthy game followed, so
            # he carried the injury into the offseason instead of rehabbing in it.
            finish = out["team"].map(team_last).fillna(season_last)
            out["ended_hurt"] = ((out["last_week"] >= (finish - 1))
                                 & ~out["last_injury"].isin(_TRANSIENT))
            out = out.drop(columns=["team"])

    # `rosters` is WEEKLY (a player goes ACT -> DEV -> INA across a season), so
    # take his LAST known row. any_value() here would pick an arbitrary week and
    # report a status he's long since left.
    status = con.sql(
        "select player_id, status from ("
        "  select gsis_id as player_id, status, row_number() over ("
        "    partition by gsis_id order by week desc, status) rn"
        "  from rosters where season = ? and gsis_id is not null) where rn = 1",
        params=[target_season],
    ).df()
    status = status[status["status"].isin(_INACTIVE_STATUS)]
    out = out.merge(status, on="player_id", how="outer")
    out["status"] = out["status"].map(_INACTIVE_STATUS)
    # A player can be known only from the roster (on IR, no report rows), so the
    # injury columns may never have been created -- pin the schema either way.
    out = out.merge(_live_status(con, target_season), on="player_id", how="outer")
    out = out.reindex(columns=["player_id", "weeks_out", "last_injury", "last_week",
                               "last_round", "ended_hurt", "status",
                               "live_code", "live_body", "live_note", "news_date"])
    for c in ("weeks_out", "last_week"):
        out[c] = out[c].astype("Float64")
    out["ended_hurt"] = out["ended_hurt"].astype("boolean").fillna(False).astype(bool)
    return out.reset_index(drop=True)


# The starting five. Both depth-chart formats label them the same way.
_OL = ("LT", "LG", "C", "RG", "RT")
# Two is where it starts to matter -- see line_context's docstring.
_OL_THRESHOLD = 2


def _ol_starters(con, season: int) -> pd.DataFrame:
    """The five projected starting linemen per team.

    Depth charts changed format: 2019-2024 are weekly rows keyed on
    `depth_position`/`depth_team`/`club_code`, 2025+ are dated snapshots keyed on
    `pos_abb`/`pos_rank`/`team`. Read whichever this season has.
    """
    ol = ", ".join(f"'{p}'" for p in _OL)
    return con.sql(f"""
        select distinct coalesce(team, club_code) as team, gsis_id
        from depth_charts
        where season = ? and gsis_id is not null and (
            (depth_team = '1' and depth_position in ({ol}))
         or (pos_rank = 1 and pos_abb in ({ol})))
    """, params=[season]).df()


def line_context(target_season: int, con=None) -> pd.DataFrame:
    """Per team: starting offensive linemen who are compromised, and who they are.

    Linemen never appear in `weekly` (ingest keeps skill positions only), but they
    decide whether a backfield has anywhere to run. `injuries` and `depth_charts`
    DO carry every position, so the unit is recoverable even though the box score
    isn't.

    MEASURED, 3,182 team-weeks 2019-2024, each team compared against its OWN
    season average so team quality cancels out:

        starting OL ruled Out |  0     1      2      3
        team RB pts vs usual  | +0.03 +0.33  -3.72  -4.65

    So it is a THRESHOLD, not a gradient -- losing one lineman costs nothing
    measurable, losing two costs a backfield ~3.8 PPR points a game (t = -3.84,
    95% CI [-5.8, -1.9]), and it replicates in both halves of the era (-3.3 in
    2019-21, -4.4 in 2022-24). A plain correlation reads -0.03 and would have
    thrown the whole thing away.

    Two related things measured as nothing and are deliberately NOT here:
      * OL continuity (how many starters return) -- r = -0.06 against RB point
        change over 192 team-seasons, non-monotone, sign backwards.
      * Opposing defenders out -- the gradient looks right (+3.5 pts at 2 out)
        but it flips sign across halves of the era (-1.2 then +6.8), so it is
        not a finding.

    Preseason caveat: in July almost nobody is on IR, so this is mostly driven by
    linemen who ended last season hurt. It earns its keep in-season.
    """
    con = con or connect()
    try:
        ol = _ol_starters(con, target_season)
    except duckdb.CatalogException:
        return pd.DataFrame(columns=["team", "ol_out", "ol_names"])
    if ol.empty:
        return pd.DataFrame(columns=["team", "ol_out", "ol_names"])

    avail = availability_context(target_season, con).set_index("player_id")
    hurt = ol["gsis_id"].map(avail["ended_hurt"]).astype("boolean").fillna(False).astype(bool)
    gone = ol["gsis_id"].map(avail["status"]).notna()
    # A lineman on IR/PUP/suspended TODAY counts; merely Questionable does not
    # (most Questionable players start, and one man down measured as no effect
    # anyway -- a soft designation must not push a team over the threshold).
    now = ol["gsis_id"].map(avail["live_code"]).isin(LIVE_SEVERE)
    ol = ol[hurt.values | gone.values | now.values]
    if ol.empty:
        return pd.DataFrame(columns=["team", "ol_out", "ol_names"])

    names = con.sql(
        "select distinct gsis_id, min(full_name) as nm from rosters "
        "where season = ? group by gsis_id", params=[target_season]).df()
    ol = ol.merge(names, on="gsis_id", how="left")
    by_team = (ol.groupby("team", as_index=False)
               .agg(ol_out=("gsis_id", "size"),
                    ol_names=("nm", lambda s: ", ".join(sorted(x for x in s if isinstance(x, str))))))
    # The measured finding is a THRESHOLD: one lineman down is noise (+0.03 vs the
    # team's usual), only two or more costs the backfield (~3.8 PPR pts/game). So a
    # compromised line means `_OL_THRESHOLD`+ out -- surfacing a single injury
    # would flag something the data says is nothing.
    return by_team[by_team["ol_out"] >= _OL_THRESHOLD].reset_index(drop=True)


def player_context(target_season: int, rules: ScoringRules = PPR, con=None) -> pd.DataFrame:
    """Situation context for every projectable player -- the room he's in.

    The season model sees a player's own prior stats plus a few team flags; it
    can't show you WHY a number might be wrong. This does, for veterans and
    rookies alike:

      * moved      -- changed teams since last season (new offense, new role)
      * blocked_by -- the best OTHER player at his position on his team, by last
                      season's points. Empty means he leads the room.
      * vacated_fp -- production at his position that left the team (opportunity)
      * depth_rank -- preseason depth-chart spot (1 = starter)
      * pass_rate  -- team's pass share of plays; scheme caps the whole room
      * new_coach  -- the team changed head coach

    Context only. These are shown next to the projection, never folded into it
    (measured worse than the model alone -- see rookie_context's note).
    """
    con = con or connect()
    prior = target_season - 1
    agg, ts = _season_agg(con, rules), _team_season(con)

    last = agg[agg["season"] == prior][["player_id", "player", "position", "fp"]]
    then = ts[ts["season"] == prior][["player_id", "team"]].rename(columns={"team": "prior_team"})
    now = ts[ts["season"] == target_season][["player_id", "team"]]
    if now.empty:                     # target-season rosters not ingested yet
        return pd.DataFrame(columns=["player_id", "team", "moved", "blocked_by",
                                     "blocked_by_fp", "vacated_fp", "depth_rank",
                                     "pass_rate", "new_coach"])

    # Who is in each room now, ranked by what they did last season.
    room = now.merge(last[["player_id", "player", "position", "fp"]], on="player_id", how="left")
    room["fp"] = room["fp"].fillna(0.0)
    room = room.dropna(subset=["position"])
    best = room.sort_values("fp", ascending=False).groupby(["team", "position"])
    top1 = best.head(1)[["team", "position", "player", "fp"]].rename(
        columns={"player": "top_player", "fp": "top_fp"})
    top2 = (best.head(2).groupby(["team", "position"]).tail(1)[["team", "position", "player", "fp"]]
            .rename(columns={"player": "second_player", "fp": "second_fp"}))
    room = room.merge(top1, on=["team", "position"], how="left").merge(
        top2, on=["team", "position"], how="left")
    # A player isn't blocked by himself: if he IS the top of the room, the man
    # behind him is irrelevant -- report nobody.
    is_top = room["player"].fillna("") == room["top_player"].fillna("")
    room["blocked_by"] = np.where(is_top, None, room["top_player"])
    room["blocked_by_fp"] = np.where(is_top, 0.0, room["top_fp"]).round(1)

    # Production that left each room since last season.
    p2 = last.merge(then, on="player_id", how="left").merge(
        now.rename(columns={"team": "team_now"}), on="player_id", how="left")
    gone = p2[p2["prior_team"] != p2["team_now"]]
    vac = (gone.groupby(["prior_team", "position"], as_index=False)["fp"].sum()
           .rename(columns={"prior_team": "team", "fp": "vacated_fp"}))
    out = room.merge(vac, on=["team", "position"], how="left")
    out["vacated_fp"] = out["vacated_fp"].fillna(0.0).round(1)

    out = out.merge(then, on="player_id", how="left")
    out["moved"] = (out["prior_team"].notna() & (out["prior_team"] != out["team"]))
    out["depth_rank"] = out["player_id"].map(_depth_rank(con, target_season))
    out = out.merge(_team_pass_rate(con, prior), on="team", how="left")

    coach = _team_coach(con)
    cur_c = coach[coach["season"] == target_season][["team", "coach"]]
    old_c = coach[coach["season"] == prior][["team", "coach"]].rename(columns={"coach": "coach_prev"})
    cc = cur_c.merge(old_c, on="team", how="left")
    cc["new_coach"] = cc["coach"].ne(cc["coach_prev"]) & cc["coach_prev"].notna()
    out = out.merge(cc[["team", "new_coach"]], on="team", how="left")
    out["new_coach"] = out["new_coach"].astype("boolean").fillna(False).astype(bool)

    out = out.merge(availability_context(target_season, con), on="player_id", how="left")
    out["ended_hurt"] = out["ended_hurt"].astype("boolean").fillna(False).astype(bool)

    # The line only measured for the backfield (RBs lose ~3.8 pts/game once two
    # starters are down; QBs showed nothing), so it rides only on RB rows rather
    # than decorating everyone with a number that means nothing for them.
    line = line_context(target_season, con)
    out = out.merge(line, on="team", how="left")
    is_rb = out["position"].eq("RB") if "position" in out.columns else False
    out["ol_out"] = out["ol_out"].where(is_rb).fillna(0).astype(int)
    out["ol_names"] = out["ol_names"].where(is_rb)

    cols = ["player_id", "team", "prior_team", "moved", "blocked_by", "blocked_by_fp",
            "vacated_fp", "depth_rank", "pass_rate", "new_coach",
            "weeks_out", "last_injury", "last_week", "last_round", "ended_hurt", "status",
            "live_code", "live_body", "live_note", "news_date", "ol_out", "ol_names"]
    return out[cols].drop_duplicates("player_id").reset_index(drop=True)
