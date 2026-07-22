"""Web-layer input validation, cache bounding, and prop parsing.

FastAPI/pydantic are optional deps (the `web` extra), so these skip cleanly in
CI where they aren't installed -- same pattern as the data-lake integration
tests. They guard the hardening added after the audit: bounded request fields
(no `range(2019, 2_000_000_000)` blowups) and a capped per-config cache.
"""

import pytest

pytest.importorskip("fastapi")

from pydantic import ValidationError  # noqa: E402

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


def test_draft_accepts_custom_rules_past_validation():
    from fastapi.testclient import TestClient

    from ffdata.web import app
    c = TestClient(app)
    # No data lake here, so it fails building the board -- but it must get PAST
    # scoring validation (not "bad scoring") when custom rules are supplied.
    r = c.post("/api/draft", json={"season": 2024, "teams": 10, "scoring": "custom",
                                   "rules": {"reception": 1.0, "pass_td": 6.0},
                                   "lineup": {"starters": {"QB": 1}, "flex": 1, "superflex": 1}}).json()
    assert r["ok"] is False and r["error"] != "bad scoring"


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
    r = c.post("/api/keepers", json={"season": 2099, "teams": 12,
                                     "keepers": [["Ja'Marr Chase", 40], ["Josh Allen", 5]]}).json()
    assert r["ok"]
    ks = {k["player"]: k for k in r["keepers"]}
    assert ks["Ja'Marr Chase"]["surplus"] == 15      # 55 - 40
    assert ks["Josh Allen"]["surplus"] == 15         # 20 - 5

    assert c.post("/api/keepers", json={"season": 2099, "keepers": []}).json()["error"].startswith("No valid")
    assert c.post("/api/keepers", json={"season": 2099, "scoring": "nope",
                                        "keepers": [["x", 1]]}).json()["error"] == "bad scoring"

    # Trade: totals per side + a verdict (diff beyond the "roughly even" band).
    t = c.post("/api/trade", json={"season": 2099, "teams": 12,
                                   "side_a": ["Ja'Marr Chase"], "side_b": ["Josh Allen"]}).json()
    assert t["ok"]
    assert t["side_a"]["auction"] == 55 and t["side_b"]["auction"] == 20
    assert t["diff"] == 35 and "Side A" in t["verdict"]
    # Close values fall in the even band.
    even = c.post("/api/trade", json={"season": 2099, "side_a": ["Ja'Marr Chase"],
                                      "side_b": ["Bijan Robinson"]}).json()
    assert even["verdict"] == "roughly even"      # 55 vs 50, within $5
    assert c.post("/api/trade", json={"season": 2099}).json()["error"].startswith("Add players")


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

    r = c.post("/api/compare", json={"season": 2099,
                                     "players": ["Ja'Marr Chase", "CeeDee Lamb", "Josh Allen"]}).json()
    assert r["ok"]
    by = {p["player"]: p for p in r["players"]}
    assert by["Ja'Marr Chase"]["overall_rank"] == 1 and by["Ja'Marr Chase"]["position_rank"] == 1
    assert by["CeeDee Lamb"]["overall_rank"] == 4 and by["CeeDee Lamb"]["position_rank"] == 2  # WR2
    assert r["best_value"] == "Ja'Marr Chase"           # highest VOR
    assert c.post("/api/compare", json={"season": 2099,
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
    r = c.post("/api/games", json={"season": 2099, "week": 15}).json()
    assert r["ok"] and r["games"][0]["total_lean"] == "over" and r["games"][0]["ml_lean"] == "KC"
    assert c.post("/api/games", json={"season": 2099, "week": 99}).status_code == 422


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

    r = c.post("/api/dynasty", json={"season": 2099, "teams": 12, "years": 4}).json()
    assert r["ok"] and r["players"][0]["dynasty_value"] == 180.0
    assert r["players"][0]["age"] == 24
    # drafted filter applies
    r2 = c.post("/api/dynasty", json={"season": 2099, "drafted": ["Young Stud"]}).json()
    assert all(p["player"] != "Young Stud" for p in r2["players"])
    # out-of-range knobs rejected by pydantic
    assert c.post("/api/dynasty", json={"season": 2099, "years": 99}).status_code == 422


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
