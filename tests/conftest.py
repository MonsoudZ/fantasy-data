"""Shared fixtures and skip logic for the ff-data test suite.

Unit tests build tiny synthetic frames and need no network. The integration
test needs a real data lake and skips automatically when one isn't present, so
`pytest` is green on a fresh clone and in CI.
"""

import pandas as pd
import pytest

from ffdata.db import RAW
from ffdata.features import USAGE_COLS

requires_data_lake = pytest.mark.skipif(
    not (RAW / "weekly").exists(),
    reason="no data lake -- run `python -m ffdata.cli` to enable integration tests",
)


def weekly_row(**overrides) -> dict:
    """A weekly-stats row with every scoring/usage column defaulted to 0."""
    row = {c: 0.0 for c in USAGE_COLS}
    row.update({
        "player_id": "00-0000000", "position": "WR", "season": 2024, "week": 1,
        "recent_team": "KC", "opponent_team": "LV",
        # raw stat columns score() reads
        "passing_yards": 0.0, "passing_tds": 0.0, "interceptions": 0.0,
        "rushing_yards": 0.0, "rushing_tds": 0.0, "receptions": 0.0,
        "receiving_yards": 0.0, "receiving_tds": 0.0,
        "rushing_fumbles_lost": 0.0, "receiving_fumbles_lost": 0.0,
        "sack_fumbles_lost": 0.0, "passing_2pt_conversions": 0.0,
        "rushing_2pt_conversions": 0.0, "receiving_2pt_conversions": 0.0,
        "special_teams_tds": 0.0,
    })
    row.update(overrides)
    return row


@pytest.fixture
def one_player_weekly() -> pd.DataFrame:
    """One WR across four weeks with known fantasy points 10/20/30/40."""
    fps = [10.0, 20.0, 30.0, 40.0]
    rows = []
    for wk, fp in enumerate(fps, start=1):
        # receiving_yards * 0.1 == fp, so score() would reproduce it, but we set
        # fp directly for rolling tests; usage rolling reads the fp column.
        rows.append(weekly_row(week=wk, fp=fp, receiving_yards=fp * 10))
    return pd.DataFrame(rows)
