"""Saved-league / saved-team persistence store. Pure stdlib + a temp file -- CI-safe."""

import pytest

from ffdata.store import (
    League, Team, delete_league, delete_team, get_league, get_team,
    list_leagues, list_teams, save_league, save_team,
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


# --- saved teams ---

@pytest.fixture
def teams(tmp_path):
    return tmp_path / "teams.json"


def test_team_roundtrips_and_normalizes_roster(teams):
    save_team(Team(name="My Squad", season=2025, scoring="half", projector="neural",
                   roster={"QB": ["Josh Allen"], "WR": ["Ja'Marr Chase", "A.J. Brown"]}),
              path=teams)
    got = get_team("my squad", path=teams)
    assert got is not None and got.projector == "neural" and got.scoring == "half"
    # Missing positions are filled in; only the four skill slots are kept.
    assert set(got.roster) == {"QB", "RB", "WR", "TE"}
    assert got.roster["QB"] == ["Josh Allen"] and got.roster["RB"] == []
    assert got.roster["WR"] == ["Ja'Marr Chase", "A.J. Brown"]


def test_team_listing_delete_and_isolation_from_leagues(teams, tmp_path):
    save_team(Team(name="B", season=2025), path=teams)
    save_team(Team(name="A", season=2025), path=teams)
    assert [t.name for t in list_teams(path=teams)] == ["A", "B"]
    # Teams live in their own file -- saving a team doesn't create leagues.
    assert list_leagues(path=tmp_path / "leagues.json") == []
    assert delete_team("a", path=teams) is True
    assert [t.name for t in list_teams(path=teams)] == ["B"]


@pytest.mark.parametrize("bad", [
    {"name": "", "season": 2025},                        # empty name
    {"name": "x", "season": 2025, "scoring": "superflex"},
    {"name": "x", "season": 2025, "projector": "quantum"},  # bad projector
    {"name": "x", "season": 1800},
])
def test_team_validation_rejects_bad(teams, bad):
    with pytest.raises(ValueError):
        save_team(Team(**bad), path=teams)
