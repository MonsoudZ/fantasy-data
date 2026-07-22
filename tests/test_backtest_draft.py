"""Draft-and-win backtest engine: snake draft, replay, schedule, playoffs, and
the orchestrator's lift mechanic -- all on synthetic frames (no data lake).

These prove the *simulation* is correct and leak-free-by-construction (the draft
never sees the weekly points). The realism of the *numbers* a real run prints is
a separate question, gated on the projections feeding it (flagged in the module).
"""

import numpy as np
import pandas as pd

from ffdata.backtest_draft import (
    best_week_total, playoffs, round_robin, run_backtest, run_snake_draft,
    simulate_season, snake_order, standings,
)


def test_snake_order_snakes_each_round():
    assert snake_order(3, 2) == [0, 1, 2, 2, 1, 0]
    assert snake_order(4, 1) == [0, 1, 2, 3]


def _board(names, positions):
    return [{"player": n, "position": p} for n, p in zip(names, positions)]


def test_snake_draft_respects_limits_and_never_double_drafts():
    names = [f"Player {c}{d}" for c in "ABCDEFGH" for d in "abcde"]   # 40 distinct
    pos = (["QB", "RB", "WR", "TE"] * 10)
    board = _board(names, pos)
    rosters = run_snake_draft([board, board, board, board], rounds=6,
                              limits={"QB": 2, "RB": 3, "WR": 3, "TE": 2})
    assert all(len(r) == 6 for r in rosters)
    # No player on two teams.
    picked = [p["player"] for r in rosters for p in r]
    assert len(picked) == len(set(picked))
    # Positional caps respected.
    for r in rosters:
        counts = {}
        for p in r:
            counts[p["position"]] = counts.get(p["position"], 0) + 1
        assert counts.get("QB", 0) <= 2 and counts.get("RB", 0) <= 3


def test_best_week_total_starts_the_top_eligible_per_slot():
    roster = _board(["Josh Allen", "Bijan Robinson", "Breece Hall", "Jamarr Chase",
                     "Puka Nacua", "Mike Evans", "Trey McBride", "Backup Back"],
                    ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB"])
    pts = {"josh allen": 25, "bijan robinson": 20, "breece hall": 18, "jamarr chase": 22,
           "puka nacua": 15, "mike evans": 10, "trey mcbride": 12, "backup back": 30}
    # Slots QB,RB,RB,WR,WR,WR,TE,FLEX -> the 30-pt "Backup Back" (RB) takes FLEX.
    # 25+20+18+22+15+10+12 + 30 = 152
    assert best_week_total(roster, pts) == 152.0


def test_round_robin_pairings_are_disjoint():
    for n in (4, 5, 8, 12):
        for pairs in round_robin(n, 14):
            flat = [t for pr in pairs for t in pr]
            assert len(flat) == len(set(flat))          # no team twice in a week


def test_standings_rank_by_wins_then_points():
    # 2 teams, 2 weeks; team 0 wins both, so it's first.
    scores = np.array([[20.0, 22.0], [10.0, 30.0]])     # wk1: 0>1, wk2: 1>0
    sched = [[(0, 1)], [(0, 1)]]
    table = standings(scores, sched)
    assert table[0]["wins"] == 1 and table[1]["wins"] == 1
    # tie on wins -> points-for breaks it: team1 has 40 vs team0's 42 -> team0 first.
    assert table[0]["team"] == 0


def test_playoffs_highest_scorer_advances():
    scores = np.zeros((4, 18))
    for t in range(4):
        scores[t, 14:] = (4 - t) * 10                    # seed 0 scores most
    assert playoffs([0, 1, 2, 3], scores, [14, 15]) == 0


def test_simulate_season_champion_is_a_playoff_seed():
    names = [f"Guy {c}{d}" for c in "ABCDEFGHIJ" for d in "ab"]        # 20 players
    pos = (["QB", "RB", "WR", "TE"] * 5)
    rosters = run_snake_draft([_board(names, pos)] * 4, rounds=5)
    wk = pd.DataFrame([{"player": n, "position": p, "week": w, "fp": 10.0}
                       for w in range(1, 18) for n, p in zip(names, pos)])
    res = simulate_season(rosters, wk, reg_weeks=14, playoff_teams=4)
    assert res["champion"] in res["seeds"]
    assert res["scores"].shape == (4, 17)


def test_run_backtest_detects_a_better_board(monkeypatch):
    import ffdata.backtest_draft as bt
    import ffdata.draft as draft

    # 40 players, balanced across positions; `val` is both the board rank key and
    # the (constant) weekly points, so a team that drafts high-val players wins.
    names, pos, val = [], [], {}
    ladders = {"QB": [30, 22, 16, 9], "RB": [29, 27, 25, 23, 14, 11, 8, 5] * 2,
               "WR": [28, 26, 24, 22, 13, 10, 7, 4] * 2, "TE": [20, 14, 9, 5]}
    for P, ladder in ladders.items():
        for i, v in enumerate(ladder):
            n = f"{P} {chr(97+i)}"
            names.append(n)
            pos.append(P)
            val[n] = float(v)
    good = pd.DataFrame({"player": names, "position": pos,
                         "proj": [val[n] for n in names], "vor": [val[n] for n in names],
                         "auction": [val[n] for n in names]}).sort_values("vor", ascending=False)

    monkeypatch.setattr(draft, "draft_board", lambda *a, **k: good)
    # Naive board = worst-first, so naive drafters take busts.
    monkeypatch.setattr(bt, "_naive_board",
                        lambda con, s, r: [{"player": n, "position": p}
                                           for n, p in sorted(zip(names, pos), key=lambda x: val[x[0]])])
    wk = pd.DataFrame([{"player": n, "position": p, "week": w, "fp": val[n]}
                       for w in range(1, 18) for n, p in zip(names, pos)])
    monkeypatch.setattr(bt, "_actual_weekly", lambda con, s, r: wk)

    out = run_backtest(2024, sims=12, teams=4, rounds=8, con=object(), seed=1)
    # Drafting the value board beats drafting worst-first: more points, better finish.
    assert out["ours"]["mean_points"] > out["naive"]["mean_points"]
    assert out["ours"]["mean_finish"] <= out["naive"]["mean_finish"]
    assert out["title_lift"] >= 0
    assert out["fair_title_rate"] == 0.25
