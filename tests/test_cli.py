"""Pure CLI/ingest helpers: season parsing and the season-rollover rule."""

import datetime as dt

import pytest

from ffdata.cli import parse_seasons
from ffdata.ingest import current_nfl_season


def test_parse_seasons_range_list_and_single():
    assert parse_seasons("2019-2022") == [2019, 2020, 2021, 2022]
    assert parse_seasons("2020,2023") == [2020, 2023]
    assert parse_seasons("2024") == [2024]


def test_parse_seasons_rejects_inverted_range():
    with pytest.raises(ValueError):
        parse_seasons("2025-2019")


def test_current_nfl_season_rolls_over_in_september():
    assert current_nfl_season(dt.date(2025, 8, 31)) == 2024   # preseason -> prior year
    assert current_nfl_season(dt.date(2025, 9, 1)) == 2025    # season starts ~Sept
