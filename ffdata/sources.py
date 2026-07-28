"""Registry of nflverse data sources.

Every dataset is a parquet (or csv) file published as a GitHub release asset
by the nflverse project. We pull them directly -- no wrapper library, no API
keys, and data is available the moment nflverse's nightly jobs publish it.
"""

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
NFLDATA = "https://github.com/nflverse/nfldata/raw/master/data"

# Datasets keyed by name. `url` is a format string taking `season`.
# `seasonal=False` means one file covers all seasons.
SOURCES = {
    # One row per player per week: rushing/receiving/passing volume and
    # efficiency, target share, air yards, plus precomputed fantasy points.
    # nflverse migrated this asset from `player_stats/player_stats_{season}`
    # to `stats_player/stats_player_week_{season}` (the old path stopped
    # publishing new seasons). The new file bundles all positions and renames
    # a couple of columns, so it is normalized on ingest (see NORMALIZERS).
    "weekly": {
        "url": f"{NFLVERSE}/stats_player/stats_player_week_{{season}}.parquet",
        "seasonal": True,
    },
    # Full play-by-play (~48k rows/season, 380+ cols). Only needed once you
    # start engineering features like red-zone touches or EPA splits.
    "pbp": {
        "url": f"{NFLVERSE}/pbp/play_by_play_{{season}}.parquet",
        "seasonal": True,
    },
    # Official injury reports (practice status, game status) per week.
    "injuries": {
        "url": f"{NFLVERSE}/injuries/injuries_{{season}}.parquet",
        "seasonal": True,
    },
    # Offense/defense/ST snap counts and percentages per player per game.
    "snap_counts": {
        "url": f"{NFLVERSE}/snap_counts/snap_counts_{{season}}.parquet",
        "seasonal": True,
    },
    # Week-by-week rosters: team, position, depth, status.
    "rosters": {
        "url": f"{NFLVERSE}/weekly_rosters/roster_weekly_{{season}}.parquet",
        "seasonal": True,
    },
    # Lee Sharpe's games file: every game since 1999 with final scores,
    # Vegas spread/total/moneyline, rest days, roof, surface. One file.
    "schedules": {
        "url": f"{NFLDATA}/games.csv",
        "seasonal": False,
    },
    # NFL draft results (round, overall pick, team, player ids, position) for
    # every draft. One all-seasons file. Powers the rookie draft-capital model
    # (draft.py): rookies have no prior NFL season, so their projection keys off
    # where they were drafted.
    "draft_picks": {
        "url": f"{NFLVERSE}/draft_picks/draft_picks.parquet",
        "seasonal": False,
    },
    # Depth charts per season -- published in the preseason, so a rookie's spot
    # on the chart (starter vs buried) is known before any games are played.
    "depth_charts": {
        "url": f"{NFLVERSE}/depth_charts/depth_charts_{{season}}.parquet",
        "seasonal": True,
    },
    # Next Gen Stats: player-tracking metrics not derivable from the box score
    # (separation, cushion, air-yards share, CPOE, rush yards over expected).
    # One all-seasons file per stat type; week 0 rows are season aggregates.
    "ngs_receiving": {
        "url": f"{NFLVERSE}/nextgen_stats/ngs_receiving.parquet",
        "seasonal": False,
    },
    "ngs_passing": {
        "url": f"{NFLVERSE}/nextgen_stats/ngs_passing.parquet",
        "seasonal": False,
    },
    "ngs_rushing": {
        "url": f"{NFLVERSE}/nextgen_stats/ngs_rushing.parquet",
        "seasonal": False,
    },
    # Pro Football Reference advanced stats, per player per week. Carries signal
    # absent from the box score: pass pressure/sacks/blitzes, dropped passes,
    # broken tackles, yards before/after contact. Keyed by pfr_player_id.
    "pfr_pass": {
        "url": f"{NFLVERSE}/pfr_advstats/advstats_week_pass_{{season}}.parquet",
        "seasonal": True,
    },
    "pfr_rec": {
        "url": f"{NFLVERSE}/pfr_advstats/advstats_week_rec_{{season}}.parquet",
        "seasonal": True,
    },
    "pfr_rush": {
        "url": f"{NFLVERSE}/pfr_advstats/advstats_week_rush_{{season}}.parquet",
        "seasonal": True,
    },
}

# Minimum contracts for columns the shipped feature/model paths actually read.
# These deliberately avoid exact schemas: nflverse can add columns freely, but a
# rename/removal of a load-bearing column must fail ingestion instead of silently
# weakening a model. ``any_of`` covers documented schema-era alternatives.
SCHEMA_CONTRACTS = {
    "weekly": {
        "required": {
            "player_id", "player_display_name", "position", "season", "week",
            "season_type", "recent_team", "opponent_team", "passing_yards",
            "passing_tds", "rushing_yards", "rushing_tds", "receptions",
            "receiving_yards", "receiving_tds",
        },
    },
    "pbp": {
        "required": {
            "game_id", "season", "week", "play_type", "yardline_100", "epa",
            "defteam", "pass", "rush", "sack", "qb_hit", "receiver_player_id",
            "rusher_player_id", "passer_player_id",
        },
    },
    "injuries": {
        "required": {
            "season", "week", "gsis_id", "position", "report_status", "practice_status",
        },
    },
    "snap_counts": {
        "required": {"season", "week", "pfr_player_id", "position", "offense_pct"},
    },
    "rosters": {
        "required": {
            "season", "team", "position", "status", "full_name", "gsis_id", "pfr_id",
        },
    },
    "schedules": {
        "required": {
            "game_id", "season", "game_type", "week", "away_team", "home_team",
            "away_score", "home_score", "spread_line", "total_line", "roof", "temp", "wind",
        },
    },
    "draft_picks": {
        "required": {"season", "round", "pick", "team", "gsis_id", "position"},
        "any_of": ({"pfr_player_name", "full_name", "player_name", "player"},),
    },
    "depth_charts": {
        "required": {"season", "gsis_id"},
        "any_of": (
            {"team", "club_code", "recent_team"},
            {"position", "pos_abb"},
            {"full_name", "player_name", "football_name"},
        ),
    },
    "ngs_receiving": {
        "required": {
            "season", "week", "player_gsis_id", "avg_separation", "avg_cushion",
            "percent_share_of_intended_air_yards", "avg_yac_above_expectation",
            "catch_percentage",
        },
    },
    "ngs_passing": {
        "required": {
            "season", "week", "player_gsis_id", "completion_percentage_above_expectation",
            "avg_time_to_throw", "aggressiveness",
        },
    },
    "ngs_rushing": {
        "required": {
            "season", "week", "player_gsis_id", "rush_yards_over_expected_per_att",
            "efficiency", "percent_attempts_gte_eight_defenders",
        },
    },
    "pfr_pass": {
        "required": {
            "season", "week", "pfr_player_id", "passing_bad_throw_pct",
            "times_pressured_pct", "times_sacked",
        },
    },
    "pfr_rec": {
        "required": {
            "season", "week", "pfr_player_id", "receiving_broken_tackles",
            "receiving_drop_pct", "receiving_rat",
        },
    },
    "pfr_rush": {
        "required": {
            "season", "week", "pfr_player_id", "rushing_yards_before_contact_avg",
            "rushing_yards_after_contact_avg", "rushing_broken_tackles",
        },
    },
}
