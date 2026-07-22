"""Feature layer: turn raw weekly stats into a leak-free modeling table.

Roadmap step 2. Every feature here answers one question -- "what did we know
*before* kickoff of week W?" -- so nothing peeks at week-W outcomes:

  * rolling usage    -- a player's trailing form and opportunity (targets,
                        shares, carries, EPA, fantasy points) over prior games
  * opponent defense -- fantasy points this week's opponent has allowed to the
                        player's position, entering the week
  * Vegas implied    -- the market's expected points for the player's team and
                        opponent, derived from schedules spread/total lines

Output is one row per player-week: the scored fantasy points (`fp`) as the
modeling target, plus `<stat>_r<N>`, `def_fp_allowed_r<N>`, and
`team_implied_total` / `opp_implied_total` feature columns.

    from ffdata.features import build_features
    feats = build_features(seasons=[2023, 2024])
    #  -> train X = feats[FEATURE_COLS], y = feats["fp"]

Leakage guard: every trailing feature is shifted so week W sees only weeks < W.
Vegas lines are set pre-game, so they are used as-is (no shift needed).
"""

from __future__ import annotations

import pandas as pd

from .db import connect
from .scoring import PPR, ScoringRules, score

# Skill positions worth modeling. DST/K live in different datasets.
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

# Usage / efficiency stats rolled into trailing averages. `fp` is added by
# score() and `snap_pct` is merged from snap_counts before rolling.
USAGE_COLS = [
    "fp",  # fantasy points -- its own strongest trailing predictor
    "targets", "receptions", "receiving_yards", "receiving_air_yards",
    "target_share", "air_yards_share", "wopr", "racr", "receiving_epa",
    "carries", "rushing_yards", "rushing_epa",
    "attempts", "passing_yards", "passing_tds", "passing_epa",
    "snap_pct",  # offensive snap share -- a leading indicator of usage
]

# Injury-report signals. Unlike usage, an injury report is known *before*
# kickoff, so these describe the current week directly (no trailing shift).
INJURY_FEATURES = ["inj_on_report", "inj_status", "practice_dnp", "practice_limited"]

# Next Gen Stats tracking metrics (game outcomes -> rolled trailing like usage).
# A player only has the metrics for their role; the rest stay NaN.
#
# FINDING: off by default because it does not help. On a 2024 GBM walk-forward,
# adding these 22 features slightly *worsened* MAE/RMSE/rank. Two reasons: NGS is
# sparse (10-35% populated by position) so mostly-NaN columns dilute, and the
# metrics (separation, air-yards share, YAC-over-expected) are largely
# correlated re-encodings of usage/efficiency the box score already provides --
# not orthogonal information. Consistent with the neural finding that the
# residual is irreducible outcome variance. Kept opt-in (`include_ngs=True`) as
# reusable infrastructure for future experiments (e.g. position-specific models
# where coverage is dense).
NGS_COLS = [
    "ngs_separation", "ngs_cushion", "ngs_ay_share", "ngs_yac_oe", "ngs_catch_pct",
    "ngs_cpoe", "ngs_time_to_throw", "ngs_aggressiveness",
    "ngs_ryoe_att", "ngs_rush_eff", "ngs_box8",
]

# PFR advanced metrics absent from the box score (game outcomes -> trailing).
# FINDING: off by default (`include_extra=True` to enable). On a 2024 GBM
# walk-forward, PFR advanced + weather together were a wash (MAE +0.007, RMSE
# -0.004, rank +0.0005 -- all within noise). Even genuinely orthogonal signals
# (pressure, drops, broken tackles, wind) don't lower the floor: the predictable
# part is already captured by usage/efficiency, and the residual is outcome
# variance. Fourth confirmation of the irreducible floor (after neural, stacking,
# NGS). Kept as opt-in infrastructure.
PFR_COLS = [
    "pfr_bad_throw_pct", "pfr_pressured_pct", "pfr_sacked",       # QB protection
    "pfr_brk_tkl_rec", "pfr_drop_pct", "pfr_rec_rat",             # receiving
    "pfr_ybc_avg", "pfr_yac_avg", "pfr_brk_tkl_rush",             # rushing
]
# Weather is known pre-game, so used for the current week (no trailing shift).
WEATHER_FEATURES = ["wind", "temp", "is_dome"]

# Red-zone opportunity from play-by-play (volume in scoring position -> trailing).
# Opportunity, not efficiency: TD scoring is high-variance, but red-zone volume
# is its leading indicator, so this is the most orthogonal thing pbp offers.
# FINDING (`include_pbp=True`): a wash on a 2024 GBM walk-forward (MAE +0.005,
# RMSE +0.009). rz_targets correlates 0.25 with fp standalone, but that signal
# is already carried by target/carry volume -- red-zone touches track total
# touches. Fifth floor confirmation. Opt-in infrastructure.
PBP_COLS = ["rz_targets", "rz_carries", "i10_targets", "i10_carries", "rz_pass_att"]

# Opponent-quality matchup metrics from pbp -- the "good player vs a specific
# weak defense" signal. Quality-adjusted (EPA) and split by pass vs rush, plus
# pass-rush pressure. Trailing per defense, merged onto a player by opponent.
# FINDING (`include_matchup=True`): a wash overall and WORSE for QBs (MAE +0.040)
# on 2024. The matchup intuition is right, but the model already captures it via
# def_fp_allowed and especially team_implied_total -- the Vegas line IS the
# matchup already priced in, and the market's forecast beats our noisy trailing
# defensive EPA. Sixth floor confirmation. Opt-in infrastructure.
MATCHUP_BASE = ["def_pass_epa", "def_rush_epa", "def_pressure"]


def _rolling_usage(weekly: pd.DataFrame, windows: tuple[int, ...],
                   cols: list[str] | None = None) -> pd.DataFrame:
    """Per-player trailing means of `cols`, shifted to exclude week W."""
    cols = cols or USAGE_COLS
    df = weekly.sort_values(["player_id", "season", "week"]).copy()
    grp = df.groupby("player_id", sort=False)[cols]
    for n in windows:
        # shift(1) drops the current week; rolling then averages only prior games
        rolled = grp.transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        df[[f"{c}_r{n}" for c in cols]] = rolled
    return df


def _opponent_defense(weekly: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """Trailing fantasy points each defense allowed to each position.

    Returns a (opponent_team, position, season, week) table whose `def_fp_allowed_rN`
    columns describe how that defense played *before* week W. Merge onto a player
    row via the player's `opponent_team` + `position`.
    """
    # Points a defense (opponent_team) surrendered to a position in one week.
    allowed = (
        weekly.groupby(["opponent_team", "position", "season", "week"], as_index=False)["fp"]
        .sum()
        .rename(columns={"fp": "fp_allowed"})
        .sort_values(["opponent_team", "position", "season", "week"])
    )
    grp = allowed.groupby(["opponent_team", "position"], sort=False)["fp_allowed"]
    for n in windows:
        allowed[f"def_fp_allowed_r{n}"] = grp.transform(
            lambda s: s.shift(1).rolling(n, min_periods=1).mean()
        )
    keep = ["opponent_team", "position", "season", "week"] + [
        f"def_fp_allowed_r{n}" for n in windows
    ]
    return allowed[keep]


def _implied_totals(schedules: pd.DataFrame) -> pd.DataFrame:
    """Per-team implied point totals from Vegas spread + total lines.

    `spread_line > 0` means the home team is favored, so
        home implied = total/2 + spread/2,  away implied = total/2 - spread/2.
    Returns one row per team per game with the team's and opponent's implied
    totals (known pre-game, so no leakage shift).
    """
    s = schedules.dropna(subset=["spread_line", "total_line"]).copy()
    half_total = s["total_line"] / 2.0
    half_spread = s["spread_line"] / 2.0
    home = pd.DataFrame({
        "season": s["season"], "week": s["week"], "team": s["home_team"],
        "team_implied_total": half_total + half_spread,
        "opp_implied_total": half_total - half_spread,
        "team_spread": s["spread_line"], "game_total": s["total_line"], "is_home": 1,
    })
    away = pd.DataFrame({
        "season": s["season"], "week": s["week"], "team": s["away_team"],
        "team_implied_total": half_total - half_spread,
        "opp_implied_total": half_total + half_spread,
        "team_spread": -s["spread_line"], "game_total": s["total_line"], "is_home": 0,
    })
    return pd.concat([home, away], ignore_index=True)


def _load_snap_pct(con, seasons: list[int]) -> pd.DataFrame:
    """Per player-week offensive snap share, keyed by gsis player_id.

    snap_counts carries only pfr_player_id, so `rosters` (which has both
    gsis_id and pfr_id) bridges it to weekly's player_id.
    """
    where = f"and s.season in ({','.join(str(int(x)) for x in seasons)})" if seasons else ""
    return con.sql(f"""
        with xwalk as (
            select season, pfr_id, min(gsis_id) as gsis_id
            from rosters
            where pfr_id is not null and gsis_id is not null
            group by season, pfr_id
        )
        select x.gsis_id as player_id, s.season, s.week,
               max(s.offense_pct) as snap_pct
        from snap_counts s
        join xwalk x on s.season = x.season and s.pfr_player_id = x.pfr_id
        where true {where}
        group by 1, 2, 3
    """).df()


def _load_injuries(con, seasons: list[int]) -> pd.DataFrame:
    """Per player-week injury-report signals, keyed by gsis player_id.

    Pre-game info, so used for the current week. `report_status` becomes an
    ordinal severity; practice participation becomes DNP / limited flags.
    """
    where = f"and season in ({','.join(str(int(x)) for x in seasons)})" if seasons else ""
    return con.sql(f"""
        select gsis_id as player_id, season, week,
               max((report_status in ('Out','Doubtful','Questionable'))::int) as inj_on_report,
               max(case report_status when 'Out' then 3 when 'Doubtful' then 2
                    when 'Questionable' then 1 else 0 end) as inj_status,
               max((practice_status like 'Did Not%%')::int) as practice_dnp,
               max((practice_status like 'Limited%%')::int) as practice_limited
        from injuries
        where gsis_id is not null {where}
        group by 1, 2, 3
    """).df()


def _load_nextgen(con, seasons: list[int]) -> pd.DataFrame:
    """Per player-week Next Gen Stats, keyed by gsis player_id.

    Three stat-type files (receiving/passing/rushing) outer-joined; week 0 rows
    (season aggregates) are excluded. Only tracking metrics that aren't already
    in `weekly` are kept, prefixed `ngs_`.
    """
    where = f"and season in ({','.join(str(int(x)) for x in seasons)})" if seasons else ""
    rec = con.sql(f"""
        select player_gsis_id as player_id, season, week,
               avg_separation as ngs_separation, avg_cushion as ngs_cushion,
               percent_share_of_intended_air_yards as ngs_ay_share,
               avg_yac_above_expectation as ngs_yac_oe, catch_percentage as ngs_catch_pct
        from ngs_receiving where week >= 1 {where}
    """).df()
    pas = con.sql(f"""
        select player_gsis_id as player_id, season, week,
               completion_percentage_above_expectation as ngs_cpoe,
               avg_time_to_throw as ngs_time_to_throw, aggressiveness as ngs_aggressiveness
        from ngs_passing where week >= 1 {where}
    """).df()
    rus = con.sql(f"""
        select player_gsis_id as player_id, season, week,
               rush_yards_over_expected_per_att as ngs_ryoe_att, efficiency as ngs_rush_eff,
               percent_attempts_gte_eight_defenders as ngs_box8
        from ngs_rushing where week >= 1 {where}
    """).df()
    keys = ["player_id", "season", "week"]
    return rec.merge(pas, on=keys, how="outer").merge(rus, on=keys, how="outer")


def _load_pfr(con, seasons: list[int]) -> pd.DataFrame:
    """Per player-week PFR advanced metrics, keyed by gsis via the roster crosswalk."""
    where = f"and s.season in ({','.join(str(int(x)) for x in seasons)})" if seasons else ""
    xwalk = "with xw as (select season, pfr_id, min(gsis_id) gsis_id from rosters " \
            "where pfr_id is not null and gsis_id is not null group by season, pfr_id)"
    pas = con.sql(f"""{xwalk}
        select x.gsis_id as player_id, s.season, s.week,
               passing_bad_throw_pct as pfr_bad_throw_pct,
               times_pressured_pct as pfr_pressured_pct, times_sacked as pfr_sacked
        from pfr_pass s join xw x on s.season=x.season and s.pfr_player_id=x.pfr_id
        where true {where}""").df()
    rec = con.sql(f"""{xwalk}
        select x.gsis_id as player_id, s.season, s.week,
               receiving_broken_tackles as pfr_brk_tkl_rec,
               receiving_drop_pct as pfr_drop_pct, receiving_rat as pfr_rec_rat
        from pfr_rec s join xw x on s.season=x.season and s.pfr_player_id=x.pfr_id
        where true {where}""").df()
    rus = con.sql(f"""{xwalk}
        select x.gsis_id as player_id, s.season, s.week,
               rushing_yards_before_contact_avg as pfr_ybc_avg,
               rushing_yards_after_contact_avg as pfr_yac_avg,
               rushing_broken_tackles as pfr_brk_tkl_rush
        from pfr_rush s join xw x on s.season=x.season and s.pfr_player_id=x.pfr_id
        where true {where}""").df()
    keys = ["player_id", "season", "week"]
    return pas.merge(rec, on=keys, how="outer").merge(rus, on=keys, how="outer")


def _load_pbp_redzone(con, seasons: list[int]) -> pd.DataFrame:
    """Per player-week red-zone opportunity counts from play-by-play (gsis ids)."""
    sw = f"and season in ({','.join(str(int(x)) for x in seasons)})" if seasons else ""
    rec = con.sql(f"""
        select receiver_player_id as player_id, season, week,
               sum((yardline_100 <= 20)::int) as rz_targets,
               sum((yardline_100 <= 10)::int) as i10_targets
        from pbp where play_type='pass' and receiver_player_id is not null {sw}
        group by 1, 2, 3""").df()
    rush = con.sql(f"""
        select rusher_player_id as player_id, season, week,
               sum((yardline_100 <= 20)::int) as rz_carries,
               sum((yardline_100 <= 10)::int) as i10_carries
        from pbp where play_type='run' and rusher_player_id is not null {sw}
        group by 1, 2, 3""").df()
    passer = con.sql(f"""
        select passer_player_id as player_id, season, week,
               sum((yardline_100 <= 20)::int) as rz_pass_att
        from pbp where play_type='pass' and passer_player_id is not null {sw}
        group by 1, 2, 3""").df()
    keys = ["player_id", "season", "week"]
    return rec.merge(rush, on=keys, how="outer").merge(passer, on=keys, how="outer")


def _def_matchup(con, seasons: list[int], windows: tuple[int, ...]) -> pd.DataFrame:
    """Trailing opponent-defense quality (pass/rush EPA allowed + pressure rate).

    Aggregated per defense per week from pbp, then shifted+rolled so it reflects
    the opponent's form *entering* week W, and merged onto a player by opponent.
    """
    sw = f"and season in ({','.join(str(int(x)) for x in seasons)})" if seasons else ""
    agg = con.sql(f"""
        select defteam as opponent_team, season, week,
               avg(case when pass=1 then epa end) as def_pass_epa,
               avg(case when rush=1 then epa end) as def_rush_epa,
               (sum(sack) + sum(qb_hit)) * 1.0
                 / nullif(sum(case when pass=1 then 1 else 0 end), 0) as def_pressure
        from pbp where defteam is not null {sw}
        group by 1, 2, 3
    """).df().sort_values(["opponent_team", "season", "week"])
    grp = agg.groupby("opponent_team", sort=False)
    for n in windows:
        for c in MATCHUP_BASE:
            agg[f"{c}_r{n}"] = grp[c].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    keep = ["opponent_team", "season", "week"] + [f"{c}_r{n}" for n in windows for c in MATCHUP_BASE]
    return agg[keep]


def _load_weather(con) -> pd.DataFrame:
    """Per team-game weather from schedules; domes are calm and climate-controlled."""
    return con.sql("""
        with g as (
            select season, week, home_team as team, wind, temp, roof from schedules
            union all
            select season, week, away_team as team, wind, temp, roof from schedules
        )
        select season, week, team,
               (roof in ('dome','closed'))::int as is_dome,
               case when roof in ('dome','closed') then 0 else wind end as wind,
               case when roof in ('dome','closed') then 70 else temp end as temp
        from g
    """).df()


def feature_columns(windows: tuple[int, ...] = (3, 5), include_ngs: bool = False,
                    include_extra: bool = False, include_pbp: bool = False,
                    include_matchup: bool = False) -> list[str]:
    """Names of the model-input columns build_features() produces."""
    roll_cols = (USAGE_COLS + (NGS_COLS if include_ngs else [])
                 + (PFR_COLS if include_extra else []) + (PBP_COLS if include_pbp else []))
    cols = [f"{c}_r{n}" for n in windows for c in roll_cols]
    cols += [f"def_fp_allowed_r{n}" for n in windows]
    cols += ["team_implied_total", "opp_implied_total", "team_spread", "game_total", "is_home"]
    cols += INJURY_FEATURES
    cols += WEATHER_FEATURES if include_extra else []
    if include_matchup:
        cols += [f"{c}_r{n}" for n in windows for c in MATCHUP_BASE]
    return cols


def build_features(
    seasons: list[int] | None = None,
    rules: ScoringRules = PPR,
    windows: tuple[int, ...] = (3, 5),
    positions: tuple[str, ...] = SKILL_POSITIONS,
    include_ngs: bool = False,
    include_extra: bool = False,
    include_pbp: bool = False,
    include_matchup: bool = False,
    con=None,
) -> pd.DataFrame:
    """Assemble the leak-free player-week modeling table.

    Args:
        seasons:     seasons to include (default: everything in the lake).
        rules:       scoring config for the `fp` target column (default PPR).
        windows:     trailing-average windows, in games, for rolling features.
        positions:   positions to keep (default skill positions).
        include_ngs: also roll Next Gen Stats metrics (off by default -- see
                     NGS_COLS; it doesn't improve accuracy).
        con:         an existing DuckDB connection; one is opened if omitted.
    """
    con = con or connect()
    where = "where season_type = 'REG'"
    if seasons:
        where += f" and season in ({','.join(str(int(s)) for s in seasons)})"
    weekly = con.sql(f"select * from weekly {where}").df()
    if positions:
        weekly = weekly[weekly["position"].isin(positions)]

    # Snap share (trailing usage) and injury report (current-week) signals.
    snaps = _load_snap_pct(con, seasons or [])
    weekly = weekly.merge(snaps, on=["player_id", "season", "week"], how="left")
    injuries = _load_injuries(con, seasons or [])
    weekly = weekly.merge(injuries, on=["player_id", "season", "week"], how="left")
    for c in INJURY_FEATURES:
        weekly[c] = weekly[c].fillna(0)  # unlisted == not on the report

    roll_cols = list(USAGE_COLS)
    if include_ngs:  # opt-in: measured to not help, kept as infrastructure
        ngs = _load_nextgen(con, seasons or [])
        weekly = weekly.merge(ngs, on=["player_id", "season", "week"], how="left")
        roll_cols += NGS_COLS
    if include_extra:  # opt-in experiment: PFR advanced metrics + weather
        pfr = _load_pfr(con, seasons or [])
        weekly = weekly.merge(pfr, on=["player_id", "season", "week"], how="left")
        roll_cols += PFR_COLS
    if include_pbp:  # opt-in experiment: red-zone opportunity from play-by-play
        rz = _load_pbp_redzone(con, seasons or [])
        weekly = weekly.merge(rz, on=["player_id", "season", "week"], how="left")
        for c in PBP_COLS:
            weekly[c] = weekly[c].fillna(0)  # played but no red-zone touch == 0
        roll_cols += PBP_COLS

    weekly = score(weekly, rules, col="fp")

    df = _rolling_usage(weekly, windows, roll_cols)
    defense = _opponent_defense(weekly, windows)
    df = df.merge(defense, on=["opponent_team", "position", "season", "week"], how="left")

    if include_matchup:  # quality-adjusted opponent defense (pass/rush EPA, pressure)
        matchup = _def_matchup(con, seasons or [], windows)
        df = df.merge(matchup, on=["opponent_team", "season", "week"], how="left")

    schedules = con.sql("select season, week, home_team, away_team, spread_line, total_line from schedules").df()
    implied = _implied_totals(schedules)
    df = df.merge(
        implied, left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "team"], how="left",
    ).drop(columns=["team"])

    if include_extra:  # weather is pre-game -> current-week feature (no shift)
        weather = _load_weather(con)
        df = df.merge(weather, left_on=["season", "week", "recent_team"],
                      right_on=["season", "week", "team"], how="left").drop(columns=["team"])

    return df.sort_values(["season", "week", "player_id"]).reset_index(drop=True)
