"""Optimizer slot-filling and win-probability logic, on a deterministic sampler."""

import numpy as np
import pandas as pd

from ffdata.optimize import LineupOptimizer, DEFAULT_SLOTS, _ELIGIBLE, _norm, _load_names, _match


class _FakeSampler:
    """Every draw equals the projection -> deterministic totals for testing."""
    def sample(self, position, pred, n):
        return np.full(n, float(pred))


class _FakeCorrSampler:
    """Joint sampler: deterministic constant draws, matrix aligned to rows."""
    def sample(self, players, n):
        return np.array([np.full(n, float(p)) for p in players["pred"]])


class _FakeSim:
    sampler = _FakeSampler()
    csampler = None


class _FakeSimCorr:
    sampler = _FakeSampler()
    csampler = _FakeCorrSampler()


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


def test_tournament_returns_valid_lineup_and_never_lowers_the_ceiling():
    pool = _pool().assign(recent_team="KC", opponent_team="LV")
    res = LineupOptimizer(_FakeSimCorr(), n_sims=200).optimize_tournament(pool, quantile=0.9)
    assert len(res["optimal"]["lineup"]) == len(DEFAULT_SLOTS)
    # Hill-climb starts at max-points, so the ceiling can only improve.
    assert res["optimal"]["ceiling"] >= res["points"]["ceiling"]
    assert isinstance(res["optimal"]["stacks"], list)


def test_optimize_uses_the_joint_correlated_sampler_when_available():
    pool = _pool().assign(recent_team="KC", opponent_team="LV")
    opp = pd.DataFrame([("O", "QB", 50, "SF", "SEA")],
                       columns=["player_display_name", "position", "pred", "recent_team", "opponent_team"])
    res = LineupOptimizer(_FakeSimCorr(), n_sims=100).optimize(pool, opp, correlated=True)
    assert len(res["optimal_lineup"]) == len(DEFAULT_SLOTS)
    assert 0.0 <= res["optimal_win_prob"] <= 1.0


# --- CLI helpers ---

def test_norm_ignores_case_punctuation_and_suffixes():
    assert _norm("Patrick Mahomes II") == "patrick mahomes"
    assert _norm("Ja'Marr Chase") == "jamarr chase"
    assert _norm("A.J. Brown") == "aj brown"
    assert _norm("Michael Pittman Jr.") == "michael pittman"


def test_load_names_reads_lines_and_skips_header(tmp_path):
    f = tmp_path / "roster.csv"
    f.write_text("player\nJosh Allen\nJa'Marr Chase\n\nBijan Robinson\n")
    assert _load_names(str(f)) == ["Josh Allen", "Ja'Marr Chase", "Bijan Robinson"]


def test_match_resolves_loosely_and_reports_misses():
    board = pd.DataFrame({
        "player_display_name": ["Josh Allen", "Ja'Marr Chase", "A.J. Brown"],
        "position": ["QB", "WR", "WR"], "pred": [22.0, 18.0, 15.0]})
    matched, missing = _match(["josh allen", "AJ Brown", "Nobody Here"], board)
    assert set(matched["player_display_name"]) == {"Josh Allen", "A.J. Brown"}
    assert missing == ["Nobody Here"]
