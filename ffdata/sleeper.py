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
_SKILL = ("QB", "RB", "WR", "TE")            # positions the VOR model ranks
_ROSTER_POS = ("QB", "RB", "WR", "TE", "K", "DEF")   # positions we roster/start


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
        # Kicker: Sleeper splits 0-19/20-29/30-39 (all typically 3); use the
        # 30-39 value as the 0-39 tier. 50p is the 50+ tier; xpm the extra point.
        fg_0_39=float(s.get("fgm_30_39", 3.0)),
        fg_40_49=float(s.get("fgm_40_49", 4.0)),
        fg_50_plus=float(s.get("fgm_50p", 5.0)),
        pat=float(s.get("xpm", 1.0)),
        fg_miss=float(s.get("fgmiss", 0.0)),
        # Team defense. Points-allowed tiers aren't taken from Sleeper yet (we use
        # the standard ladder); the counting-stat weights are mapped here.
        dst_sack=float(s.get("sack", 1.0)),
        dst_int=float(s.get("int", 2.0)),
        dst_fumble_rec=float(s.get("fum_rec", 2.0)),
        dst_td=float(s.get("def_td", 6.0)),
        dst_safety=float(s.get("safe", 2.0)),
        dst_block=float(s.get("blk_kick", 2.0)),
    )


def map_roster(rosters: list, user_id, players: dict) -> dict:
    """The named roster owned by `user_id`, by position (incl. K and DEF).

    Team defenses in Sleeper are keyed by team abbreviation with position 'DEF';
    we store them as '<TEAM> DST' to match the projection board's naming.
    """
    roster = {p: [] for p in _ROSTER_POS}
    mine = next((r for r in rosters if str(r.get("owner_id")) == str(user_id)), None)
    if not mine:
        return roster
    for pid in (mine.get("players") or []):
        info = players.get(str(pid)) or {}
        pos = info.get("position")
        if pos == "DEF":
            team = info.get("team") or str(pid)
            roster["DEF"].append(f"{team} DST")
        elif pos in roster and info.get("full_name"):
            roster[pos].append(info["full_name"])
    return roster


def map_roster_positions(positions: list) -> dict:
    """Sleeper `roster_positions` -> starting-lineup slots.

    Returns {"starters": {QB/RB/WR/TE/K/DEF counts}, "flex": n, "superflex": n}.
    Bench (BN), IR, taxi, and IDP slots are ignored. SUPER_FLEX/OP are
    QB-eligible; DEF/DST and K are counted so standard leagues start them.
    """
    starters = {p: 0 for p in _ROSTER_POS}
    flex = superflex = 0
    for tok in positions or []:
        t = str(tok).upper()
        if t in ("DST", "D/ST"):        # normalize Sleeper's defense token to DEF
            t = "DEF"
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


# --------------------------------------------------------------------------- #
# Live availability feed -> the lake
# --------------------------------------------------------------------------- #

# Sleeper's `injury_status` codes. This is where suspensions actually live --
# NOT in the top-level `status` field, which only ever reads Active/Inactive.
LIVE_STATUS = {
    "Sus": "suspended",
    "DNR": "not reporting",
    "IR": "on injured reserve",
    "PUP": "on PUP",
    "NA": "on non-football injury",
    "COV": "on the COVID list",
    "Out": "out",
    "Doubtful": "doubtful",
    "Questionable": "questionable",
}
# Nothing above this is a hard absence; Questionable/Doubtful are soft.
LIVE_SEVERE = ("Sus", "DNR", "IR", "PUP", "NA", "Out")
# Sleeper asks callers to hit /players/nfl at most once a day. Well inside that.
LIVE_TTL_HOURS = 12
_LIVE_DATASET = "sleeper_status"


def norm_name(s: str | None) -> str:
    """Squash a display name to a join key.

    Sleeper's `gsis_id` is only populated for ~16% of rostered skill players (and
    some of those carry stray whitespace), so name+position is the real join --
    it matches 88% with zero collisions across the 2026 skill pool.
    """
    s = (s or "").lower()
    for suffix in (" jr", " sr", " ii", " iii", " iv", " v"):
        if s.endswith(suffix) or s.endswith(suffix + "."):
            s = s[: -len(suffix)]
    return "".join(ch for ch in s if ch.isalpha())


def live_rows(players: dict) -> list[dict]:
    """Sleeper's player blob -> availability rows. Pure; `players` is the JSON."""
    import datetime as dt

    out = []
    for v in (players or {}).values():
        if not isinstance(v, dict):
            continue
        # Every player on an NFL roster, NOT just the ones we rank: a suspended
        # left tackle never scores a fantasy point but still costs the backfield
        # behind him (see draft.line_context). Sleeper ships literal "Duplicate
        # Player" placeholder rows -- drop those.
        if not v.get("team") or not v.get("position"):
            continue
        if v.get("full_name") == "Duplicate Player":
            continue
        gsis = (v.get("gsis_id") or "").strip() or None
        updated = v.get("news_updated")
        if updated:
            updated = dt.datetime.fromtimestamp(updated / 1000).date().isoformat()
        out.append({
            "gsis_id": gsis,
            "name_key": norm_name(v.get("full_name")),
            "position": v.get("position"),
            "team": v.get("team"),
            "live_code": v.get("injury_status") or None,
            "live_body": v.get("injury_body_part") or None,
            "live_note": v.get("injury_notes") or None,
            "news_date": updated,
        })
    return out


def refresh_live_status(client: "SleeperClient | None" = None, force: bool = False,
                        log=print) -> int:
    """Pull Sleeper's live feed into the lake as the `sleeper_status` view.

    This is the ONLY thing here that touches the network, and it is an explicit
    ingest step -- the draft board reads the cached parquet and never fetches, so
    it stays fast and works offline. Returns the row count written (0 if the
    cache was still fresh).
    """
    import time

    import pandas as pd

    from .db import RAW

    dest = RAW / _LIVE_DATASET / f"{_LIVE_DATASET}.parquet"
    if dest.exists() and not force:
        age_h = (time.time() - dest.stat().st_mtime) / 3600
        if age_h < LIVE_TTL_HOURS:
            log(f"  skip  {_LIVE_DATASET} (fresh, {age_h:.1f}h old)")
            return 0

    rows = live_rows((client or SleeperClient()).players())
    df = pd.DataFrame(rows, columns=["gsis_id", "name_key", "position", "team",
                                     "live_code", "live_body", "live_note", "news_date"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    flagged = int(df["live_code"].notna().sum())
    log(f"  ok    {_LIVE_DATASET}: {len(df):,} players, {flagged} flagged")
    return len(df)
