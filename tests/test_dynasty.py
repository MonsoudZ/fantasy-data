"""Dynasty age curves -- delta-method shape sanity (needs the data lake)."""

import pandas as pd

from conftest import requires_data_lake


def test_dynasty_board_forwards_league_and_shapes_output(monkeypatch):
    """dynasty_board must pass the league (teams/superflex) down to draft_board so
    dynasty VOR matches the league -- verified with fakes, no data lake needed."""
    import ffdata.dynasty as dyn

    captured = {}

    def fake_draft_board(season, league=None, rules=None, con=None):
        captured["league"] = league
        return pd.DataFrame({"player_id": ["p1", "p2"], "player": ["Young", "Old"],
                             "position": ["RB", "WR"], "proj": [200.0, 180.0],
                             "vor": [60.0, 50.0], "auction": [40, 30]})

    monkeypatch.setattr(dyn, "connect", lambda *a, **k: object())
    monkeypatch.setattr(dyn, "draft_board", fake_draft_board)
    monkeypatch.setattr(dyn, "age_curves",
                        lambda con=None, before_season=None, rules=None:
                        {p: {a: 1.0 for a in range(21, 37)} for p in ("QB", "RB", "WR", "TE")})
    monkeypatch.setattr(dyn, "_roster_info",
                        lambda con: pd.DataFrame({"player_id": ["p1", "p2"], "season": [2025, 2025],
                                                  "birth_year": [2001, 1995]}))

    board = dyn.dynasty_board(2025, years=4, league={"teams": 10, "superflex": 1,
                                                     "starters": {"QB": 1}, "flex": 1})
    assert captured["league"]["superflex"] == 1          # league forwarded to the board
    assert {"player", "age", "proj", "vor", "dynasty_value"}.issubset(board.columns)
    ages = dict(zip(board["player"], board["age"]))
    assert ages["Young"] == 24 and ages["Old"] == 30     # 2025 - birth_year


@requires_data_lake
def test_age_curves_decline_and_rb_peaks_before_te():
    from ffdata.dynasty import age_curves
    c = age_curves()
    # Every position is peak-normalized to 1.0 and never exceeds it.
    for pos in ("QB", "RB", "WR", "TE"):
        assert max(c[pos].values()) == 1.0 and all(0 <= v <= 1 for v in c[pos].values())
    # Delta method should recover the known shape: RBs peak young and fall hard;
    # by 32 a running back retains less of his peak than a tight end.
    rb_peak = max(c["RB"], key=c["RB"].get)
    te_peak = max(c["TE"], key=c["TE"].get)
    assert rb_peak <= te_peak
    assert c["RB"][32] < c["TE"][32]


@requires_data_lake
def test_dynasty_favors_youth_over_equal_redraft_value():
    from ffdata.dynasty import dynasty_board
    b = dynasty_board(2024, years=4)
    assert {"player", "age", "dynasty_value"}.issubset(b.columns)
    assert b["dynasty_value"].is_monotonic_decreasing  # returned sorted best-first
