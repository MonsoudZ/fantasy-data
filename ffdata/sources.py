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
