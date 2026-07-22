"""Web-layer input validation, cache bounding, and prop parsing.

FastAPI/pydantic are optional deps (the `web` extra), so these skip cleanly in
CI where they aren't installed -- same pattern as the data-lake integration
tests. They guard the hardening added after the audit: bounded request fields
(no `range(2019, 2_000_000_000)` blowups) and a capped per-config cache.
"""

import pytest

pytest.importorskip("fastapi")

from conftest import requires_data_lake  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from ffdata.ingest import current_nfl_season  # noqa: E402

# A season that has actually been PLAYED. These tests monkeypatch the projection
# board, so the value only has to clear the "season hasn't kicked off" guard --
# derived, not a literal, so it can't go stale (see matchup.fit_seasons).
_PLAYED = current_nfl_season()

from ffdata.web import (  # noqa: E402
    DraftRequest, OptRequest, PropsRequest, _cache_put, _MAX_CACHE, _parse_props,
)


def test_optrequest_defaults_to_a_sane_season():
    r = OptRequest(week=5)
    assert 1999 <= r.season <= 2100


@pytest.mark.parametrize("kwargs", [
    {"week": 5, "season": 2_000_000_000},   # the range()-blowup vector
    {"week": 99},                            # week out of range
    {"week": 0},
    {"week": 5, "ceiling": 1.5},             # quantile must be < 1
])
def test_optrequest_rejects_out_of_range(kwargs):
    with pytest.raises(ValidationError):
        OptRequest(**kwargs)


def test_draft_and_props_requests_bound_their_fields():
    with pytest.raises(ValidationError):
        DraftRequest(teams=999)
    with pytest.raises(ValidationError):
        DraftRequest(n=100_000)
    with pytest.raises(ValidationError):
        PropsRequest(week=50)


def test_cache_put_caps_size_and_evicts_oldest():
    cache: dict = {}
    for i in range(_MAX_CACHE + 5):
        _cache_put(cache, i, i)
    assert len(cache) == _MAX_CACHE
    assert 0 not in cache                    # oldest evicted
    assert (_MAX_CACHE + 4) in cache         # newest kept
    # Re-putting an existing key doesn't grow or evict.
    existing = next(iter(cache))
    _cache_put(cache, existing, "v")
    assert len(cache) == _MAX_CACHE and cache[existing] == "v"


def test_parse_props_skips_header_and_malformed_rows():
    text = ("player,market,line,over,under\n"
            "Josh Allen,passing_yards,250.5,-110,-110\n"
            "bad,row\n")
    df = _parse_props(text)
    assert list(df["player"]) == ["Josh Allen"]
    assert df.iloc[0]["line"] == 250.5


def test_league_crud_via_api(tmp_path, monkeypatch):
    # Point the store at a temp file so the test never touches ~/.ff-data.
    monkeypatch.setenv("FFDATA_STATE", str(tmp_path / "leagues.json"))
    from fastapi.testclient import TestClient

    from ffdata.web import app
    c = TestClient(app)

    assert c.get("/api/leagues").json() == {"ok": True, "leagues": []}

    saved = c.post("/api/leagues", json={"name": "Home", "season": 2025,
                                         "scoring": "half", "teams": 10,
                                         "drafted": ["Bijan Robinson"]}).json()
    assert saved["ok"] and saved["league"]["scoring"] == "half"
    assert saved["league"]["drafted"] == ["Bijan Robinson"]

    listed = c.get("/api/leagues").json()["leagues"]
    assert len(listed) == 1 and listed[0]["name"] == "Home"

    # Store-level validation surfaces as ok:false; pydantic bounds as 422.
    assert c.post("/api/leagues", json={"name": "X", "season": 2025,
                                        "scoring": "superflex"}).json()["ok"] is False
    assert c.post("/api/leagues", json={"name": "Y", "season": 2025,
                                        "teams": 999}).status_code == 422

    assert c.post("/api/leagues/delete", json={"name": "home"}).json()["deleted"] is True
    assert c.get("/api/leagues").json()["leagues"] == []


def test_team_crud_via_api(tmp_path, monkeypatch):
    # Teams live in teams.json beside the leagues file the state var points at.
    monkeypatch.setenv("FFDATA_STATE", str(tmp_path / "leagues.json"))
    from fastapi.testclient import TestClient

    from ffdata.web import app
    c = TestClient(app)

    assert c.get("/api/teams").json() == {"ok": True, "teams": []}

    saved = c.post("/api/teams", json={"name": "Squad", "season": 2025,
                                       "scoring": "half", "projector": "neural",
                                       "roster": {"QB": ["Josh Allen"]}}).json()
    assert saved["ok"] and saved["team"]["projector"] == "neural"
    assert saved["team"]["roster"]["QB"] == ["Josh Allen"]
    assert saved["team"]["roster"]["RB"] == []          # normalized to 4 slots

    assert len(c.get("/api/teams").json()["teams"]) == 1
    assert c.post("/api/teams", json={"name": "Z", "season": 2025,
                                      "projector": "quantum"}).json()["ok"] is False
    assert c.post("/api/teams", json={"name": "W", "season": 3000}).status_code == 422
    assert c.post("/api/teams/delete", json={"name": "squad"}).json()["deleted"] is True
    assert c.get("/api/teams").json()["teams"] == []


def test_draft_rejects_an_unknown_scoring_name():
    """Validation is checked without data: a bogus scoring name is rejected
    before the board is ever built."""
    from fastapi.testclient import TestClient

    from ffdata.web import app
    r = TestClient(app).post("/api/draft", json={"season": 2024, "scoring": "quantum"}).json()
    assert r["ok"] is False and r["error"] == "bad scoring"


@requires_data_lake
def test_draft_builds_a_board_with_custom_rules():
    """Against the real lake: custom rules pass validation AND produce a board."""
    from fastapi.testclient import TestClient

    from ffdata.web import app
    r = TestClient(app).post("/api/draft", json={
        "season": 2024, "teams": 10, "scoring": "custom",
        "rules": {"reception": 1.0, "pass_td": 6.0},
        "lineup": {"starters": {"QB": 1}, "flex": 1, "superflex": 1}}).json()
    assert r["ok"] is True, r.get("error")
    assert r["players"], "custom-scoring draft board came back empty"
    top = r["players"][0]
    assert {"player", "position", "proj", "vor", "auction"} <= set(top)


def test_league_cfg_applies_imported_lineup():
    from ffdata.web import _league_cfg
    cfg = _league_cfg(10, {"starters": {"QB": 2, "RB": 2}, "flex": 1, "superflex": 1})
    assert cfg["teams"] == 10 and cfg["superflex"] == 1
    assert cfg["starters"]["QB"] == 2
    # unspecified positions fall back to the default lineup
    assert cfg["starters"]["WR"] == 3
    assert _league_cfg(12, None)["superflex"] == 0   # default has no superflex


def test_keepers_and_trades_endpoints(monkeypatch):
    import pandas as pd

    import ffdata.web as web
    board = pd.DataFrame({
        "player": ["Ja'Marr Chase", "Bijan Robinson", "Josh Allen", "CeeDee Lamb"],
        "position": ["WR", "RB", "QB", "WR"],
        "proj": [280.0, 270.0, 360.0, 260.0],
        "vor": [80.0, 70.0, 60.0, 55.0],
        "auction": [55, 50, 20, 45],
    })
    monkeypatch.setattr(web, "draft_board", lambda *a, **k: board)
    web._DRAFT.clear()

    from fastapi.testclient import TestClient
    c = TestClient(web.app)

    # Keepers: surplus = auction value - cost.
    r = c.post("/api/keepers", json={"season": _PLAYED, "teams": 12,
                                     "keepers": [["Ja'Marr Chase", 40], ["Josh Allen", 5]]}).json()
    assert r["ok"]
    ks = {k["player"]: k for k in r["keepers"]}
    assert ks["Ja'Marr Chase"]["surplus"] == 15      # 55 - 40
    assert ks["Josh Allen"]["surplus"] == 15         # 20 - 5

    assert c.post("/api/keepers", json={"season": _PLAYED, "keepers": []}).json()["error"].startswith("No valid")
    assert c.post("/api/keepers", json={"season": _PLAYED, "scoring": "nope",
                                        "keepers": [["x", 1]]}).json()["error"] == "bad scoring"

    # Trade: totals per side + a verdict (diff beyond the "roughly even" band).
    t = c.post("/api/trade", json={"season": _PLAYED, "teams": 12,
                                   "side_a": ["Ja'Marr Chase"], "side_b": ["Josh Allen"]}).json()
    assert t["ok"]
    assert t["side_a"]["auction"] == 55 and t["side_b"]["auction"] == 20
    assert t["diff"] == 35 and "Side A" in t["verdict"]
    # Close values fall in the even band.
    even = c.post("/api/trade", json={"season": _PLAYED, "side_a": ["Ja'Marr Chase"],
                                      "side_b": ["Bijan Robinson"]}).json()
    assert even["verdict"] == "roughly even"      # 55 vs 50, within $5
    assert c.post("/api/trade", json={"season": _PLAYED}).json()["error"].startswith("Add players")


def test_compare_endpoint(monkeypatch):
    import pandas as pd

    import ffdata.web as web
    board = pd.DataFrame({                               # already VOR-desc sorted
        "player": ["Ja'Marr Chase", "Bijan Robinson", "Josh Allen", "CeeDee Lamb"],
        "position": ["WR", "RB", "QB", "WR"],
        "proj": [280.0, 270.0, 360.0, 260.0],
        "vor": [80.0, 70.0, 60.0, 55.0],
        "auction": [55, 50, 20, 45],
    })
    monkeypatch.setattr(web, "draft_board", lambda *a, **k: board)
    web._DRAFT.clear()

    from fastapi.testclient import TestClient
    c = TestClient(web.app)

    r = c.post("/api/compare", json={"season": _PLAYED,
                                     "players": ["Ja'Marr Chase", "CeeDee Lamb", "Josh Allen"]}).json()
    assert r["ok"]
    by = {p["player"]: p for p in r["players"]}
    assert by["Ja'Marr Chase"]["overall_rank"] == 1 and by["Ja'Marr Chase"]["position_rank"] == 1
    assert by["CeeDee Lamb"]["overall_rank"] == 4 and by["CeeDee Lamb"]["position_rank"] == 2  # WR2
    assert r["best_value"] == "Ja'Marr Chase"           # highest VOR
    assert c.post("/api/compare", json={"season": _PLAYED,
                                        "players": ["Josh Allen"]}).json()["error"].startswith("Pick at least")


def test_games_endpoint(monkeypatch):
    import pandas as pd

    import ffdata.web as web
    board = pd.DataFrame([{
        "game": "LV @ KC", "home": "KC", "away": "LV",
        "total_line": 45.0, "pred_total": 50.0, "total_lean": "over",
        "model_over": 0.62, "mkt_over": 0.5,
        "spread_line": 3.0, "pred_margin": 7.0, "spread_lean": "KC",
        "model_home_cover": 0.58, "mkt_home_cover": 0.52,
        "model_home_win": 0.65, "mkt_home_win": 0.58, "ml_lean": "KC",
    }])
    monkeypatch.setattr(web, "game_forecasts", lambda *a, **k: board)
    web._GAMES.clear()

    from fastapi.testclient import TestClient
    c = TestClient(web.app)
    r = c.post("/api/games", json={"season": _PLAYED, "week": 15}).json()
    assert r["ok"] and r["games"][0]["total_lean"] == "over" and r["games"][0]["ml_lean"] == "KC"
    assert c.post("/api/games", json={"season": _PLAYED, "week": 99}).status_code == 422


def test_dynasty_endpoint(monkeypatch):
    import pandas as pd

    import ffdata.web as web
    dboard = pd.DataFrame({
        "player": ["Young Stud", "Aging Vet"], "position": ["RB", "WR"],
        "age": [24, 30], "proj": [230.0, 210.0], "vor": [70.0, 60.0],
        "dynasty_value": [180.0, 90.0],
    })
    monkeypatch.setattr(web, "dynasty_board", lambda *a, **k: dboard)
    web._DYN.clear()

    from fastapi.testclient import TestClient
    c = TestClient(web.app)

    r = c.post("/api/dynasty", json={"season": _PLAYED, "teams": 12, "years": 4}).json()
    assert r["ok"] and r["players"][0]["dynasty_value"] == 180.0
    assert r["players"][0]["age"] == 24
    # drafted filter applies
    r2 = c.post("/api/dynasty", json={"season": _PLAYED, "drafted": ["Young Stud"]}).json()
    assert all(p["player"] != "Young Stud" for p in r2["players"])
    # out-of-range knobs rejected by pydantic
    assert c.post("/api/dynasty", json={"season": _PLAYED, "years": 99}).status_code == 422


def test_sleeper_import_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("FFDATA_STATE", str(tmp_path / "leagues.json"))
    import ffdata.web as web
    from ffdata.store import League, Team

    rules = {"reception": 1.0, "pass_td": 6.0}
    monkeypatch.setattr(web, "list_user_leagues",
                        lambda u, s: [{"league_id": "L1", "name": "Home",
                                       "teams": 10, "scoring": "custom"}])
    monkeypatch.setattr(web, "import_league", lambda lid, u, s: (
        League(name="Home", season=s, teams=10, scoring="custom", rules=rules,
               drafted=["Josh Allen"]),
        Team(name="Home", season=s, scoring="custom", rules=rules,
             roster={"QB": ["Josh Allen"], "RB": [], "WR": [], "TE": []})))

    from fastapi.testclient import TestClient
    c = TestClient(web.app)

    listed = c.post("/api/import/sleeper/leagues",
                    json={"username": "mcsleeper", "season": 2025}).json()
    assert listed["ok"] and listed["leagues"][0]["league_id"] == "L1"

    imp = c.post("/api/import/sleeper/league",
                 json={"league_id": "L1", "username": "mcsleeper", "season": 2025}).json()
    assert imp["ok"] and imp["scoring"] == "custom"
    assert imp["drafted"] == 1 and imp["roster_size"] == 1

    # The import persisted both a saved league and a saved team.
    assert any(lg["name"] == "Home" for lg in c.get("/api/leagues").json()["leagues"])
    assert any(t["name"] == "Home" for t in c.get("/api/teams").json()["teams"])


def test_freeagents_endpoint(monkeypatch):
    import pandas as pd

    import ffdata.web as web
    board = pd.DataFrame({
        "player_display_name": ["Josh Allen", "Bijan Robinson", "Breece Hall", "Jamarr Chase",
                                "Puka Nacua", "Mike Evans", "Trey McBride",
                                "Jalen Hurts", "Nico Collins"],
        "position": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "QB", "WR"],
        "pred": [18.0, 15.0, 12.0, 14.0, 11.0, 7.0, 9.0, 25.0, 20.0],
        "recent_team": ["BUF", "ATL", "NYJ", "CIN", "LAR", "TB", "ARI", "PHI", "HOU"],
    })
    # Skip the (heavy) model fit: _board returns (sim, board); we only use board.
    monkeypatch.setattr(web, "_board", lambda *a, **k: (None, board))

    from fastapi.testclient import TestClient
    c = TestClient(web.app)
    roster = "\n".join(["Josh Allen", "Bijan Robinson", "Breece Hall", "Jamarr Chase",
                        "Puka Nacua", "Mike Evans", "Trey McBride"])

    # 1-QB league: a spare QB only helps by upgrading the QB slot.
    r = c.post("/api/freeagents", json={"season": _PLAYED, "week": 5, "roster": roster}).json()
    assert r["ok"] and r["starter_proj"] == 86.0
    ups = {u["player"]: u for u in r["upgrades"]}
    assert ups["Nico Collins"]["gain"] == 20.0          # fills empty FLEX
    assert ups["Jalen Hurts"]["gain"] == 7.0            # 25 - 18, benches Josh Allen

    # Superflex: the second QB now starts outright -> full value.
    sf = c.post("/api/freeagents", json={"season": _PLAYED, "week": 5, "roster": roster,
        "lineup": {"starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flex": 1, "superflex": 1}}).json()
    assert {u["player"]: u for u in sf["upgrades"]}["Jalen Hurts"]["gain"] == 25.0

    # An empty roster is rejected before any projection work.
    assert c.post("/api/freeagents", json={"season": _PLAYED, "week": 5}).json()["error"].startswith("Add your roster")


def test_optimize_fills_defense_and_kicker_slots(monkeypatch):
    import pandas as pd

    import ffdata.web as web
    board = pd.DataFrame({
        "player_display_name": ["Josh Allen", "Bijan Robinson", "Breece Hall", "Jamarr Chase",
                                "Puka Nacua", "Mike Evans", "Trey McBride",
                                "BUF DST", "Justin Tucker"],
        "position": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "DEF", "K"],
        "pred": [22.0, 16.0, 13.0, 18.0, 14.0, 9.0, 11.0, 8.0, 7.0],
        "recent_team": ["BUF", "ATL", "NYJ", "CIN", "LAR", "TB", "ARI", "BUF", "BAL"],
    })
    # Default (no-opponent) optimize uses the greedy fill -> no live sim needed.
    monkeypatch.setattr(web, "_board", lambda *a, **k: (None, board))

    from fastapi.testclient import TestClient
    c = TestClient(web.app)
    roster = "\n".join(board["player_display_name"])
    r = c.post("/api/optimize", json={
        "season": _PLAYED, "week": 5, "roster": roster,
        "lineup": {"starters": {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1},
                   "flex": 1, "superflex": 0}}).json()
    assert r["ok"]
    slots = {row["slot"]: row["name"] for row in r["lineup"]}
    assert slots["DEF"] == "BUF DST" and slots["K"] == "Justin Tucker"


def test_markets_endpoint_pairs_each_stat_with_its_positions():
    """Drives the props picker so a QB can't be offered `receptions`."""
    from fastapi.testclient import TestClient

    from ffdata.web import app
    d = TestClient(app).get("/api/markets").json()
    assert d["ok"]
    assert d["markets"]["passing_yards"] == ["QB"]
    assert "QB" not in d["markets"]["receptions"]
    assert set(d["markets"]["receiving_yards"]) == {"WR", "TE", "RB"}


def test_names_endpoint_rejects_bad_scoring():
    from fastapi.testclient import TestClient

    from ffdata.web import app
    d = TestClient(app).post("/api/names", json={"scoring": "nonsense", "teams": 12}).json()
    assert d["ok"] is False and "scoring" in d["error"]


@requires_data_lake
def test_names_endpoint_returns_the_whole_board_for_the_pickers():
    """The pickers need EVERY draftable player, not the top-N the board shows --
    otherwise a keeper or trade target outside the top 50 can't be selected."""
    from fastapi.testclient import TestClient

    from ffdata.web import app
    d = TestClient(app).post("/api/names", json={"season": 2024, "scoring": "ppr",
                                                "teams": 12}).json()
    assert d["ok"] is True, d.get("error")
    assert d["count"] > 300
    assert {"player", "position", "proj", "vor", "auction"} <= set(d["players"][0])
    # Sorted by value, so a picker's first hits are the players you'd actually want.
    vors = [p["vor"] for p in d["players"]]
    assert vors == sorted(vors, reverse=True)


def test_config_exposes_one_season_for_the_whole_ui():
    """There is deliberately no season picker: earlier seasons are training data,
    never something the user selects. Showing last year's number beside this
    year's advice is how you draft for a season that already happened."""
    from fastapi.testclient import TestClient

    from ffdata.ingest import upcoming_nfl_season
    from ffdata.web import app
    c = TestClient(app).get("/api/config").json()
    assert c["season"] == upcoming_nfl_season()
    assert c["draft_season"] == c["season"], "draft and in-season views must agree"
    assert c["started"] is (c["season"] <= current_nfl_season())


@pytest.mark.parametrize("path,body", [
    ("/api/players", {"week": 5}),
    ("/api/optimize", {"week": 5, "roster": "Josh Allen"}),
    ("/api/freeagents", {"week": 5, "roster": "Josh Allen"}),
    ("/api/props", {"week": 5, "lines": "Josh Allen,passing_yards,250,-110,-110"}),
])
def test_weekly_tools_say_the_season_has_not_started(path, body):
    """Before Week 1 exists there are no weekly stats. Say so, rather than dying
    on an empty frame or quietly serving last season under this season's label."""
    from fastapi.testclient import TestClient

    from ffdata.web import app
    r = TestClient(app).post(path, json={**body, "season": current_nfl_season() + 3}).json()
    assert r["ok"] is False
    assert r["not_started"] is True
    assert "hasn't kicked off" in r["error"]


def test_season_not_started_predicate():
    import datetime as dt

    from ffdata.ingest import season_not_started
    july = dt.date(2026, 7, 22)      # offseason: 2025 is the last played season
    assert season_not_started(2026, july) is True
    assert season_not_started(2025, july) is False
    september = dt.date(2026, 9, 20)  # 2026 is under way
    assert season_not_started(2026, september) is False
