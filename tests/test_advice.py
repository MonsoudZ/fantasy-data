"""Grounded advice layer: prompt assembly, the availability gate, and the
/api/advice endpoint -- all with a mocked Anthropic client (no live network)."""

import json
import types

import pytest

from ffdata import advice


class _Block:
    def __init__(self, kind, text=""):
        self.type = kind
        self.text = text


class _FakeMessages:
    """Records the last create() kwargs and returns a canned message."""
    def __init__(self, blocks):
        self._blocks = blocks
        self.last = None

    def create(self, **kwargs):
        self.last = kwargs
        return types.SimpleNamespace(content=self._blocks)


class _FakeClient:
    def __init__(self, blocks):
        self.messages = _FakeMessages(blocks)


def test_available_false_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert advice.available() is False


def test_available_true_with_key_and_sdk(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert advice.available() is True


def test_explain_grounds_prompt_and_returns_text():
    facts = {"decision": "compare",
             "players": [{"player": "Ja'Marr Chase", "vor": 80.0},
                         {"player": "CeeDee Lamb", "vor": 55.0}]}
    client = _FakeClient([_Block("thinking"), _Block("text", "Take Chase: VOR 80 vs 55.")])
    out = advice.explain("compare", facts, client=client)

    assert out == "Take Chase: VOR 80 vs 55."          # thinking block stripped
    sent = client.messages.last
    assert sent["model"] == advice.MODEL
    assert sent["thinking"] == {"type": "adaptive"}
    # No sampling / budget params (rejected on Opus 4.8).
    assert not ({"temperature", "top_p", "top_k", "budget_tokens"} & set(sent))
    # The facts are embedded verbatim as JSON so every claim is checkable.
    user = sent["messages"][0]["content"]
    assert json.dumps(facts, indent=2, sort_keys=True) in user
    assert "80" in user and "Chase" in user


def test_explain_rejects_unknown_kind():
    with pytest.raises(ValueError):
        advice.explain("nonsense", {}, client=_FakeClient([]))


def test_api_advice_off_when_unavailable(monkeypatch):
    pytest.importorskip("fastapi")
    import ffdata.web as web
    from fastapi.testclient import TestClient

    monkeypatch.setattr(web.advice, "available", lambda: False)
    c = TestClient(web.app)
    r = c.post("/api/advice", json={"season": 2099, "kind": "compare",
                                    "players": ["A", "B"]}).json()
    assert r["ok"] is False and "Advice is off" in r["error"]


def test_api_advice_grounds_on_engine_output(monkeypatch):
    pytest.importorskip("fastapi")
    import pandas as pd

    import ffdata.web as web
    from fastapi.testclient import TestClient

    board = pd.DataFrame({                               # VOR-desc sorted
        "player": ["Ja'Marr Chase", "Bijan Robinson", "Josh Allen", "CeeDee Lamb"],
        "position": ["WR", "RB", "QB", "WR"],
        "proj": [280.0, 270.0, 360.0, 260.0],
        "vor": [80.0, 70.0, 60.0, 55.0],
        "auction": [55, 50, 20, 45],
    })
    monkeypatch.setattr(web, "draft_board", lambda *a, **k: board)
    web._DRAFT.clear()

    captured = {}

    def fake_explain(kind, facts):
        captured["kind"] = kind
        captured["facts"] = facts
        return "grounded text"

    monkeypatch.setattr(web.advice, "available", lambda: True)
    monkeypatch.setattr(web.advice, "explain", fake_explain)

    c = TestClient(web.app)

    # compare -> facts carry the ranked rows + scoring context.
    r = c.post("/api/advice", json={"season": 2099, "teams": 12, "kind": "compare",
                                    "players": ["Ja'Marr Chase", "CeeDee Lamb"]}).json()
    assert r["ok"] and r["advice"] == "grounded text" and r["kind"] == "compare"
    f = captured["facts"]
    assert f["decision"] == "compare" and f["scoring"] == "ppr" and f["teams"] == 12
    names = {p["player"] for p in f["players"]}
    assert names == {"Ja'Marr Chase", "CeeDee Lamb"}

    # keeper -> surplus rows.
    c.post("/api/advice", json={"season": 2099, "teams": 12, "kind": "keeper",
                                "keepers": [["Ja'Marr Chase", 40]]}).json()
    assert captured["facts"]["decision"] == "keeper"
    assert captured["facts"]["keepers"][0]["surplus"] == 15   # 55 - 40

    # trade -> per-side totals + verdict.
    c.post("/api/advice", json={"season": 2099, "teams": 12, "kind": "trade",
                                "side_a": ["Ja'Marr Chase"], "side_b": ["Josh Allen"]}).json()
    tf = captured["facts"]
    assert tf["decision"] == "trade" and tf["diff"] == 35 and "Side A" in tf["verdict"]


def test_api_advice_validates_inputs(monkeypatch):
    pytest.importorskip("fastapi")
    import pandas as pd

    import ffdata.web as web
    from fastapi.testclient import TestClient

    board = pd.DataFrame({
        "player": ["Ja'Marr Chase", "Josh Allen"], "position": ["WR", "QB"],
        "proj": [280.0, 360.0], "vor": [80.0, 60.0], "auction": [55, 20],
    })
    monkeypatch.setattr(web, "draft_board", lambda *a, **k: board)
    web._DRAFT.clear()
    monkeypatch.setattr(web.advice, "available", lambda: True)
    monkeypatch.setattr(web.advice, "explain", lambda k, f: "x")

    c = TestClient(web.app)
    # compare needs >=2 players.
    r = c.post("/api/advice", json={"season": 2099, "kind": "compare",
                                    "players": ["Josh Allen"]}).json()
    assert r["ok"] is False and "at least 2" in r["error"]
    # unknown kind is rejected before any model call.
    r = c.post("/api/advice", json={"season": 2099, "kind": "bogus"}).json()
    assert r["ok"] is False and "unknown advice kind" in r["error"]
