"""Optimizer slot-filling and win-probability logic, on a deterministic sampler."""

import numpy as np
import pandas as pd

from ffdata.optimize import LineupOptimizer, DEFAULT_SLOTS, _ELIGIBLE


class _FakeSampler:
    """Every draw equals the projection -> deterministic totals for testing."""
    def sample(self, position, pred, n):
        return np.full(n, float(pred))


class _FakeSim:
    sampler = _FakeSampler()


def _pool():
    rows = [("QB1", "QB", 20), ("QB2", "QB", 18),
            ("RB1", "RB", 16), ("RB2", "RB", 14), ("RB3", "RB", 10),
            ("WR1", "WR", 15), ("WR2", "WR", 13), ("WR3", "WR", 12), ("WR4", "WR", 9),
            ("TE1", "TE", 11), ("TE2", "TE", 6)]
    return pd.DataFrame(rows, columns=["player_display_name", "position", "pred"])


def test_greedy_fills_every_slot_with_eligible_players():
    opt = LineupOptimizer(_FakeSim(), n_sims=50)
    lineup = opt._greedy_points(_pool())
    assert len(lineup) == len(DEFAULT_SLOTS)
    for slot, _, pos, _ in lineup:
        assert pos in _ELIGIBLE[slot]
    # No player used twice.
    names = [x[1] for x in lineup]
    assert len(set(names)) == len(names)


def test_greedy_takes_highest_projected_per_slot():
    opt = LineupOptimizer(_FakeSim(), n_sims=50)
    lineup = {slot: name for slot, name, _, _ in opt._greedy_points(_pool())}
    assert lineup["QB"] == "QB1"          # best QB
    assert lineup["TE"] == "TE1"          # best TE
    # FLEX should take the best leftover RB/WR/TE (RB3=10 vs WR4=9 vs TE2=6 -> RB3)
    assert lineup["FLEX"] == "RB3"


def test_winprob_is_deterministic_zero_or_one_with_constant_draws():
    opt = LineupOptimizer(_FakeSim(), n_sims=100)
    vecs = {"a": np.full(100, 10.0), "b": np.full(100, 5.0)}
    opp = np.full(100, 12.0)
    assert opt._winprob(["a", "b"], vecs, opp) == 1.0   # 15 > 12
    assert opt._winprob(["b"], vecs, opp) == 0.0        # 5 < 12


def test_optimize_returns_valid_lineups_and_never_worse_than_points():
    opt = LineupOptimizer(_FakeSim(), n_sims=100)
    pool = _pool()
    opp = pd.DataFrame([("O", "QB", 50)], columns=["player_display_name", "position", "pred"])
    res = opt.optimize(pool, opp)
    assert 0.0 <= res["optimal_win_prob"] <= 1.0
    # Hill-climb starts from the points lineup, so it can never end up worse.
    assert res["optimal_win_prob"] >= res["points_win_prob"]
    assert len(res["optimal_lineup"]) == len(DEFAULT_SLOTS)
