"""Sleeper import mappers, on sample JSON (no network -- runs in CI).

Validates the mapping core: scoring_settings -> ScoringRules (exact custom
scoring), rosters+players -> a named roster, draft picks -> drafted names, and
the import orchestration via an injected fake client.
"""

from ffdata.scoring import HALF_PPR, PPR, preset_name, rules_from, rules_to_dict
from ffdata.sleeper import (
    import_league, list_user_leagues, map_roster, map_roster_positions, map_scoring,
)


def test_map_roster_positions_counts_starters_flex_superflex():
    positions = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "SUPER_FLEX",
                 "BN", "BN", "K", "DEF", "IR"]
    m = map_roster_positions(positions)
    # K and DEF are now real starting slots (standard leagues); BN/IR still ignored.
    assert m["starters"] == {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1}
    assert m["flex"] == 1 and m["superflex"] == 1
    # Sleeper's DST token normalizes to DEF.
    assert map_roster_positions(["DST"])["starters"]["DEF"] == 1


# --- scoring helpers round-trip ---

def test_scoring_helpers_roundtrip():
    assert rules_from("half") == HALF_PPR
    assert rules_from(custom=rules_to_dict(PPR)) == PPR
    assert preset_name(HALF_PPR) == "half"
    assert preset_name(PPR) == "ppr"
    # Unknown keys ignored; a genuinely custom sheet reads as 'custom'.
    r = rules_from(custom={"reception": 1.0, "pass_td": 6.0, "bogus": 9})
    assert r.pass_td == 6.0 and preset_name(r) == "custom"


# --- Sleeper mappers ---

def test_map_scoring_defaults_to_ppr():
    assert preset_name(map_scoring({})) == "ppr"
    assert preset_name(map_scoring({"rec": 0.5})) == "half"
    assert preset_name(map_scoring({"rec": 0.0})) == "standard"


def test_map_scoring_keeps_custom_values():
    r = map_scoring({"rec": 1.0, "bonus_rec_te": 0.5, "pass_td": 6, "pass_int": -1})
    assert r.te_reception_bonus == 0.5
    assert r.pass_td == 6.0
    assert r.interception == -1.0        # Sleeper's `pass_int` -> our `interception`
    assert preset_name(r) == "custom"


_PLAYERS = {
    "p1": {"full_name": "Josh Allen", "position": "QB"},
    "p2": {"full_name": "Bijan Robinson", "position": "RB"},
    "p3": {"full_name": "Ja'Marr Chase", "position": "WR"},
    "p4": {"full_name": "CeeDee Lamb", "position": "WR"},
    "p5": {"full_name": "Justin Tucker", "position": "K"},
    "BUF": {"position": "DEF", "team": "BUF"},    # team defense: keyed by team, no full_name
}


def test_map_roster_groups_skill_positions_for_the_owner():
    rosters = [{"owner_id": "u123", "players": ["p1", "p2", "p3", "p5", "BUF"]},
               {"owner_id": "u999", "players": ["p4"]}]
    roster = map_roster(rosters, "u123", _PLAYERS)
    assert roster["QB"] == ["Josh Allen"]
    assert roster["RB"] == ["Bijan Robinson"]
    assert roster["WR"] == ["Ja'Marr Chase"]     # p4 belongs to the other owner
    assert roster["TE"] == []
    assert roster["K"] == ["Justin Tucker"]
    assert roster["DEF"] == ["BUF DST"]          # team defense named to match the board


class _FakeClient:
    def __init__(self, data):
        self.data = data

    def user(self, username):
        return self.data["user"]

    def user_leagues(self, uid, season):
        return self.data["user_leagues"]

    def league(self, lid):
        return self.data["league"]

    def rosters(self, lid):
        return self.data["rosters"]

    def draft_picks(self, did):
        return self.data["picks"]

    def players(self):
        return _PLAYERS


def _fixture():
    return {
        "user": {"user_id": "u123", "username": "mcsleeper"},
        "user_leagues": [{"league_id": "L1", "name": "Home", "total_rosters": 12,
                          "scoring_settings": {"rec": 0.5}, "draft_id": "D1"}],
        "league": {"league_id": "L1", "name": "Home Dynasty", "total_rosters": 10,
                   "scoring_settings": {"rec": 1.0, "bonus_rec_te": 0.5, "pass_td": 6},
                   "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "SUPER_FLEX",
                                        "DEF", "K", "BN"],
                   "draft_id": "D1"},
        "rosters": [{"owner_id": "u123", "players": ["p1", "p2", "p3", "p5", "BUF"]},
                    {"owner_id": "u999", "players": ["p4"]}],
        "picks": [{"player_id": "p1"}, {"player_id": "p4"}],
    }


def test_list_user_leagues_summarizes():
    got = list_user_leagues("mcsleeper", 2025, client=_FakeClient(_fixture()))
    assert got == [{"league_id": "L1", "name": "Home", "teams": 12, "scoring": "half"}]


def test_import_league_builds_league_and_team():
    league, team = import_league("L1", "mcsleeper", 2025, client=_FakeClient(_fixture()))

    # League: settings + exact custom scoring + drafted names.
    assert league.name == "Home Dynasty" and league.season == 2025 and league.teams == 10
    assert league.scoring == "custom"               # TE-premium + 6pt pass TD
    assert league.rules["te_reception_bonus"] == 0.5 and league.rules["pass_td"] == 6.0
    assert league.drafted == ["Josh Allen", "CeeDee Lamb"]

    # Starting-lineup shape imported too (superflex, and a DEF + K starter).
    assert league.lineup["superflex"] == 1
    assert league.lineup["starters"] == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}

    # Team: the caller's roster (incl. K and team defense), same scoring, validates.
    assert team.roster["QB"] == ["Josh Allen"]
    assert team.roster["WR"] == ["Ja'Marr Chase"]
    assert team.roster["K"] == ["Justin Tucker"]
    assert team.roster["DEF"] == ["BUF DST"]
    assert team.rules == league.rules
    team.validated()                                # roster normalizes, no raise
