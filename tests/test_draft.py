"""Draft value logic: replacement levels, VOR-ranked availability, auction split."""

import pandas as pd

from ffdata.draft import (_replacement_ranks, best_available, keeper_value, trade_value,
                          round_cost, DEFAULT_LEAGUE)


def _valued_board():
    return pd.DataFrame({
        "player": ["Stud", "Mid", "Cheap", "Bench"], "position": ["WR", "RB", "WR", "TE"],
        "proj": [300, 250, 200, 150], "vor": [150, 100, 60, 20], "auction": [70, 45, 25, 5]})


def test_keeper_surplus_ranks_bargains_first():
    kp = keeper_value(_valued_board(), [("Cheap", 10), ("Stud", 65)], cost_type="auction")
    d = dict(zip(kp["player"], kp["surplus"]))
    assert d["Cheap"] == 15 and d["Stud"] == 5      # value - cost
    assert kp["player"].iloc[0] == "Cheap"          # biggest surplus on top


def test_trade_value_totals_and_verdict():
    r = trade_value(_valued_board(), ["Stud"], ["Mid", "Cheap"])
    assert r["side_a"]["auction"] == 70 and r["side_b"]["auction"] == 70
    assert "even" in r["verdict"]


def test_round_cost_falls_with_later_rounds():
    b = _valued_board()
    assert round_cost(b, 1, teams=1) >= round_cost(b, 2, teams=1)


def test_replacement_ranks_reflect_position_depth():
    r = _replacement_ranks(DEFAULT_LEAGUE)  # 12 teams, QB1/RB2/WR3/TE1 + 1 FLEX
    assert r["QB"] == 12                     # 12 teams x 1 starting QB, no flex
    assert r["WR"] > r["RB"] > r["TE"]       # WR demand highest, then RB, then TE
    assert r["RB"] > 24 and r["WR"] > 36     # FLEX pushes the RB/WR replacement deeper


def test_replacement_ranks_scale_with_league_size():
    small = _replacement_ranks({**DEFAULT_LEAGUE, "teams": 8})
    big = _replacement_ranks({**DEFAULT_LEAGUE, "teams": 14})
    assert big["QB"] > small["QB"]           # more teams -> shallower replacement


def _board():
    return pd.DataFrame({
        "player": ["A", "B", "C", "D", "E"],
        "position": ["RB", "WR", "RB", "WR", "TE"],
        "proj": [300, 290, 250, 240, 200],
        "vor": [150, 140, 100, 90, 50],
        "auction": [60, 55, 40, 35, 20],
    })


def test_best_available_excludes_drafted_and_keeps_vor_order():
    out = best_available(_board(), drafted=["B"], n=10)
    assert list(out["player"]) == ["A", "C", "D", "E"]   # B removed, VOR order kept


def test_best_available_filters_by_position():
    out = best_available(_board(), position="RB")
    assert set(out["player"]) == {"A", "C"}


def test_best_available_matches_names_loosely():
    # Drafted list uses a different casing/punctuation than the board.
    board = _board().assign(player=["A.J. Brown", "B", "C", "D", "E"])
    out = best_available(board, drafted=["aj brown"])
    assert "A.J. Brown" not in set(out["player"])
