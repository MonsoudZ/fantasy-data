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


def _rolling_usage(weekly: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """Per-player trailing means of USAGE_COLS, shifted to exclude week W."""
    df = weekly.sort_values(["player_id", "season", "week"]).copy()
    grp = df.groupby("player_id", sort=False)[USAGE_COLS]
    for n in windows:
        # shift(1) drops the current week; rolling then averages only prior games
        rolled = grp.transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        df[[f"{c}_r{n}" for c in USAGE_COLS]] = rolled
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
            select season, pfr_id, any_value(gsis_id) as gsis_id
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


def feature_columns(windows: tuple[int, ...] = (3, 5)) -> list[str]:
    """Names of the model-input columns build_features() produces."""
    cols = [f"{c}_r{n}" for n in windows for c in USAGE_COLS]
    cols += [f"def_fp_allowed_r{n}" for n in windows]
    cols += ["team_implied_total", "opp_implied_total", "team_spread", "game_total", "is_home"]
    cols += INJURY_FEATURES
    return cols


def build_features(
    seasons: list[int] | None = None,
    rules: ScoringRules = PPR,
    windows: tuple[int, ...] = (3, 5),
    positions: tuple[str, ...] = SKILL_POSITIONS,
    con=None,
) -> pd.DataFrame:
    """Assemble the leak-free player-week modeling table.

    Args:
        seasons:   seasons to include (default: everything in the lake).
        rules:     scoring config for the `fp` target column (default PPR).
        windows:   trailing-average windows, in games, for rolling features.
        positions: positions to keep (default skill positions).
        con:       an existing DuckDB connection; one is opened if omitted.
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

    weekly = score(weekly, rules, col="fp")

    df = _rolling_usage(weekly, windows)
    defense = _opponent_defense(weekly, windows)
    df = df.merge(defense, on=["opponent_team", "position", "season", "week"], how="left")

    schedules = con.sql("select season, week, home_team, away_team, spread_line, total_line from schedules").df()
    implied = _implied_totals(schedules)
    df = df.merge(
        implied, left_on=["season", "week", "recent_team"],
        right_on=["season", "week", "team"], how="left",
    ).drop(columns=["team"])

    return df.sort_values(["season", "week", "player_id"]).reset_index(drop=True)
