"""Import a league from Sleeper.

Sleeper publishes a **public, read-only REST API** with no auth or API key
(https://docs.sleeper.com), so a user can pull their league from just a username:

    from ffdata.sleeper import list_user_leagues, import_league
    for lg in list_user_leagues("mcsleeper", 2025):
        print(lg["name"], lg["teams"], lg["scoring"])
    league, team = import_league(lg["league_id"], "mcsleeper", 2025)
    # -> a store.League (settings + scoring + drafted) and store.Team (your roster)

The pure mappers (scoring_settings -> ScoringRules, rosters+players -> a roster,
draft picks -> drafted names) are the tested core; the HTTP client is a thin
wrapper over urllib. NOTE: outbound calls need network access -- validate against
a live account before relying on it; unit tests here inject sample JSON.
"""

from __future__ import annotations

import json
import urllib.request

from .scoring import ScoringRules, preset_name, rules_to_dict
from .store import League, Team

_BASE = "https://api.sleeper.app/v1"
_UA = "ff-data-sleeper/0.1"
_SKILL = ("QB", "RB", "WR", "TE")


# --------------------------------------------------------------------------- #
# HTTP client (thin; not unit-tested -- needs the network)
# --------------------------------------------------------------------------- #

class SleeperClient:  # pragma: no cover - thin network I/O, exercised via a fake in tests
    """Minimal client over Sleeper's public endpoints. Injectable for testing."""

    def __init__(self, base: str = _BASE, timeout: int = 30):
        self.base, self.timeout = base, timeout
        self._players: dict | None = None

    def _get(self, path: str):
        req = urllib.request.Request(f"{self.base}{path}", headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def user(self, username: str) -> dict:
        return self._get(f"/user/{username}") or {}

    def user_leagues(self, user_id: str, season: int) -> list:
        return self._get(f"/user/{user_id}/leagues/nfl/{season}") or []

    def league(self, league_id: str) -> dict:
        return self._get(f"/league/{league_id}") or {}

    def rosters(self, league_id: str) -> list:
        return self._get(f"/league/{league_id}/rosters") or []

    def draft_picks(self, draft_id: str) -> list:
        return self._get(f"/draft/{draft_id}/picks") or []

    def players(self) -> dict:
        # ~5MB; fetched once per client and cached.
        if self._players is None:
            self._players = self._get("/players/nfl") or {}
        return self._players


# --------------------------------------------------------------------------- #
# Pure mappers (Sleeper JSON -> our models) -- the tested core
# --------------------------------------------------------------------------- #

def map_scoring(scoring_settings: dict) -> ScoringRules:
    """Sleeper `scoring_settings` -> a ScoringRules (exact custom scoring kept)."""
    s = scoring_settings or {}
    return ScoringRules(
        pass_yd=float(s.get("pass_yd", 0.04)),
        pass_td=float(s.get("pass_td", 4.0)),
        interception=float(s.get("pass_int", -2.0)),
        rush_yd=float(s.get("rush_yd", 0.1)),
        rush_td=float(s.get("rush_td", 6.0)),
        reception=float(s.get("rec", 1.0)),
        rec_yd=float(s.get("rec_yd", 0.1)),
        rec_td=float(s.get("rec_td", 6.0)),
        te_reception_bonus=float(s.get("bonus_rec_te", 0.0)),
        fumble_lost=float(s.get("fum_lost", -2.0)),
        two_pt=float(s.get("rec_2pt", 2.0)),
        special_teams_td=float(s.get("st_td", 6.0)),
    )


def map_roster(rosters: list, user_id, players: dict) -> dict:
    """The named skill-position roster owned by `user_id`, by position."""
    roster = {p: [] for p in _SKILL}
    mine = next((r for r in rosters if str(r.get("owner_id")) == str(user_id)), None)
    if not mine:
        return roster
    for pid in (mine.get("players") or []):
        info = players.get(str(pid)) or {}
        pos, name = info.get("position"), info.get("full_name")
        if pos in roster and name:
            roster[pos].append(name)
    return roster


def map_roster_positions(positions: list) -> dict:
    """Sleeper `roster_positions` -> starting-lineup slots for VOR.

    Returns {"starters": {QB/RB/WR/TE counts}, "flex": n, "superflex": n}. Bench
    (BN), IR, taxi, and K/DEF/IDP slots are ignored -- only the skill starters
    and flex types shape replacement level. SUPER_FLEX/OP are QB-eligible.
    """
    starters = {p: 0 for p in _SKILL}
    flex = superflex = 0
    for tok in positions or []:
        t = str(tok).upper()
        if t in starters:
            starters[t] += 1
        elif t in ("SUPER_FLEX", "SUPERFLEX", "OP"):
            superflex += 1
        elif "FLEX" in t:               # FLEX, REC_FLEX, WRRB_FLEX, ...
            flex += 1
    return {"starters": starters, "flex": flex, "superflex": superflex}


def _pick_names(picks: list, players: dict) -> list[str]:
    names = []
    for pk in picks or []:
        info = players.get(str(pk.get("player_id"))) or {}
        if info.get("full_name"):
            names.append(info["full_name"])
    return names


def _summarize(league_json: dict) -> dict:
    rules = map_scoring(league_json.get("scoring_settings") or {})
    return {"league_id": league_json.get("league_id"),
            "name": league_json.get("name") or "Sleeper league",
            "teams": int(league_json.get("total_rosters") or 0),
            "scoring": preset_name(rules)}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def list_user_leagues(username: str, season: int, client: SleeperClient | None = None) -> list[dict]:
    """Every league a username is in for `season`, summarized for a picker."""
    client = client or SleeperClient()
    uid = client.user(username).get("user_id")
    if not uid:
        return []
    return [_summarize(lg) for lg in client.user_leagues(uid, season)]


def import_league(league_id: str, username: str, season: int,
                  client: SleeperClient | None = None) -> tuple[League, Team]:
    """Pull a Sleeper league into a store.League (settings/scoring/drafted) and a
    store.Team (the caller's roster). Does not persist -- the caller saves."""
    client = client or SleeperClient()
    league_json = client.league(league_id)
    rosters = client.rosters(league_id)
    players = client.players()
    uid = client.user(username).get("user_id")

    rules = map_scoring(league_json.get("scoring_settings") or {})
    rules_dict = rules_to_dict(rules)
    label = preset_name(rules)
    name = league_json.get("name") or "Sleeper league"
    teams = int(league_json.get("total_rosters") or 12)

    picks = client.draft_picks(league_json["draft_id"]) if league_json.get("draft_id") else []
    drafted = _pick_names(picks, players)
    lineup = map_roster_positions(league_json.get("roster_positions") or [])

    league = League(name=name, season=season, teams=teams,
                    scoring=label, rules=rules_dict, drafted=drafted, lineup=lineup)
    team = Team(name=name, season=season, scoring=label, rules=rules_dict,
                roster=map_roster(rosters, uid, players))
    return league, team
