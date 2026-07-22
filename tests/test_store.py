"""Saved-league persistence store. Pure stdlib + a temp file -- runs in CI."""

import pytest

from ffdata.store import (
    League, delete_league, get_league, list_leagues, save_league,
)


@pytest.fixture
def store(tmp_path):
    return tmp_path / "leagues.json"


def test_save_then_get_roundtrips(store):
    lg = League(name="Home 12", season=2025, scoring="half", teams=10,
                drafted=["Bijan Robinson"], keepers=[["Ja'Marr Chase", 40]])
    save_league(lg, path=store)
    got = get_league("home 12", path=store)          # case-insensitive lookup
    assert got is not None
    assert got.season == 2025 and got.scoring == "half" and got.teams == 10
    assert got.drafted == ["Bijan Robinson"]
    assert got.keepers == [["Ja'Marr Chase", 40]]


def test_save_overwrites_same_name(store):
    save_league(League(name="Dynasty", season=2025, teams=12), path=store)
    save_league(League(name="dynasty", season=2026, teams=14), path=store)  # same key
    leagues = list_leagues(path=store)
    assert len(leagues) == 1
    assert leagues[0].season == 2026 and leagues[0].teams == 14


def test_list_is_sorted_and_delete_works(store):
    for n in ["Zeta", "alpha", "Mid"]:
        save_league(League(name=n, season=2025), path=store)
    assert [lg.name for lg in list_leagues(path=store)] == ["alpha", "Mid", "Zeta"]
    assert delete_league("MID", path=store) is True
    assert delete_league("nope", path=store) is False
    assert [lg.name for lg in list_leagues(path=store)] == ["alpha", "Zeta"]


def test_missing_store_is_empty_not_an_error(store):
    assert list_leagues(path=store) == []
    assert get_league("anything", path=store) is None


def test_corrupt_store_degrades_gracefully(store):
    store.write_text("{ this is not json")
    assert list_leagues(path=store) == []          # no crash on a mangled file


@pytest.mark.parametrize("bad", [
    {"name": "", "season": 2025},                  # empty name
    {"name": "x", "season": 2025, "scoring": "superflex"},  # bad scoring
    {"name": "x", "season": 2025, "teams": 99},    # teams out of range
    {"name": "x", "season": 1800},                 # season out of range
])
def test_validation_rejects_bad_leagues(store, bad):
    with pytest.raises(ValueError):
        save_league(League(**bad), path=store)


def test_writes_are_atomic_no_tmp_left_behind(store):
    save_league(League(name="A", season=2025), path=store)
    # The temp sibling used for the atomic replace must not linger.
    assert not store.with_suffix(store.suffix + ".tmp").exists()
