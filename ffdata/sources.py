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
    "weekly": {
        "url": f"{NFLVERSE}/player_stats/player_stats_{{season}}.parquet",
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
}
