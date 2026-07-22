"""Optimizer slot-filling and win-probability logic, on a deterministic sampler."""

import numpy as np
import pandas as pd

from ffdata.optimize import (
    LineupOptimizer, DEFAULT_SLOTS, _ELIGIBLE, _norm, _load_names, _match,
    free_agent_advice, slots_from_lineup,
)


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


def _assert_valid_lineup(lineup, pool):
    """A lineup [(slot, name), ...] must have no duplicate players and every
    player must be eligible for its slot -- checked on the FINAL lineup, not just
    the greedy start, so a bad hill-climb swap can't slip through."""
    pos = dict(zip(pool["player_display_name"], pool["position"]))
    names = [n for _, n in lineup]
    assert len(set(names)) == len(names), f"duplicate player in lineup: {names}"
    for slot, name in lineup:
        assert pos[name] in _ELIGIBLE[slot], f"{name} ({pos[name]}) ineligible for slot {slot}"


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
    # The hill-climbed result must still be a legal lineup (no dupes, slot-eligible).
    _assert_valid_lineup(res["optimal_lineup"], pool)
    _assert_valid_lineup(res["points_lineup"], pool)


def test_tournament_returns_valid_lineup_and_never_lowers_the_ceiling():
    pool = _pool().assign(recent_team="KC", opponent_team="LV")
    res = LineupOptimizer(_FakeSimCorr(), n_sims=200).optimize_tournament(pool, quantile=0.9)
    assert len(res["optimal"]["lineup"]) == len(DEFAULT_SLOTS)
    # Hill-climb starts at max-points, so the ceiling can only improve.
    assert res["optimal"]["ceiling"] >= res["points"]["ceiling"]
    assert isinstance(res["optimal"]["stacks"], list)
    _assert_valid_lineup(res["optimal"]["lineup"], pool)


def test_game_stack_builds_a_qb_receiver_stack():
    # Pool with a KC QB + two KC receivers, plus fillers from other games.
    rows = [("KC_QB", "QB", 22, "KC", "LV"), ("KC_WR1", "WR", 15, "KC", "LV"),
            ("KC_TE", "TE", 12, "KC", "LV"), ("LV_WR", "WR", 13, "LV", "KC"),
            ("RB_a", "RB", 16, "SF", "SEA"), ("RB_b", "RB", 15, "DAL", "NYG"),
            ("RB_c", "RB", 12, "GB", "CHI"), ("WR_x", "WR", 17, "MIA", "BUF"),
            ("WR_y", "WR", 16, "PHI", "WAS"), ("QB2", "QB", 20, "BUF", "MIA")]
    pool = pd.DataFrame(rows, columns=["player_display_name", "position", "pred",
                                       "recent_team", "opponent_team"])
    res = LineupOptimizer(_FakeSimCorr(), n_sims=200).optimize_game_stack(
        pool, quantile=0.9, stack_size=2, bringback=1)
    stack = res["optimal"]["stack"]
    assert stack is not None
    # The stack must be a QB + 2 own receivers + 1 opponent receiver.
    assert stack["qb"] == "KC_QB" and stack["team"] == "KC"
    assert "KC" in res["optimal"]["stacks"]           # QB + own receiver present
    names = [n for _, n in res["optimal"]["lineup"]]
    assert {"KC_QB", "KC_WR1", "KC_TE", "LV_WR"}.issubset(names)
    _assert_valid_lineup(res["optimal"]["lineup"], pool)


def test_optimize_uses_the_joint_correlated_sampler_when_available():
    pool = _pool().assign(recent_team="KC", opponent_team="LV")
    opp = pd.DataFrame([("O", "QB", 50, "SF", "SEA")],
                       columns=["player_display_name", "position", "pred", "recent_team", "opponent_team"])
    res = LineupOptimizer(_FakeSimCorr(), n_sims=100).optimize(pool, opp, correlated=True)
    assert len(res["optimal_lineup"]) == len(DEFAULT_SLOTS)
    assert 0.0 <= res["optimal_win_prob"] <= 1.0


# --- superflex slots ---

def test_slots_from_lineup_defaults_and_superflex():
    assert slots_from_lineup(None) == DEFAULT_SLOTS
    assert slots_from_lineup({"starters": {}, "flex": 0, "superflex": 0}) == DEFAULT_SLOTS
    sf = slots_from_lineup({"starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1},
                            "flex": 1, "superflex": 1})
    assert sf.count("SUPERFLEX") == 1 and sf.count("QB") == 1 and sf.count("FLEX") == 1
    # A SUPERFLEX slot accepts a QB (that's the whole point of a 2-QB league).
    assert "QB" in _ELIGIBLE["SUPERFLEX"]


def test_superflex_slot_lets_a_second_qb_start():
    opt = LineupOptimizer(_FakeSim(), slots=slots_from_lineup(
        {"starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flex": 1, "superflex": 1}), n_sims=50)
    lineup = opt._greedy_points(_pool())
    starters = [(s, n) for s, n, _, _ in lineup]
    # Both QBs start: QB1 in the QB slot, QB2 taking SUPERFLEX (18 > any leftover flex option).
    assert ("QB", "QB1") in starters and ("SUPERFLEX", "QB2") in starters


# --- free-agent / waiver advice ---

def _fa_board():
    rows = [("Josh Allen", "QB", 18), ("Bijan Robinson", "RB", 15), ("Breece Hall", "RB", 12),
            ("Jamarr Chase", "WR", 14), ("Puka Nacua", "WR", 11), ("Mike Evans", "WR", 7),
            ("Trey McBride", "TE", 9),
            ("Jalen Hurts", "QB", 25), ("Nico Collins", "WR", 20), ("Tank Bigsby", "RB", 6)]
    return pd.DataFrame(rows, columns=["player_display_name", "position", "pred"])


_MY = ["Josh Allen", "Bijan Robinson", "Breece Hall", "Jamarr Chase",
       "Puka Nacua", "Mike Evans", "Trey McBride"]


def test_free_agent_gain_is_marginal_lineup_improvement_not_raw_projection():
    res = free_agent_advice(_fa_board(), _MY, slots=DEFAULT_SLOTS)
    assert res["starter_proj"] == 86.0     # 7 starters, FLEX empty (roster is 7 players)
    ups = {u["player"]: u for u in res["upgrades"]}
    # Nico (WR 20) fills the empty FLEX -> full 20 gain, benches nobody.
    assert ups["Nico Collins"]["gain"] == 20.0 and ups["Nico Collins"]["replaces"] is None
    # In a 1-QB lineup a spare QB only helps by replacing the worse QB: 25 - 18 = 7.
    assert ups["Jalen Hurts"]["gain"] == 7.0 and ups["Jalen Hurts"]["replaces"] == "Josh Allen"


def test_free_agent_superflex_values_a_second_qb_fully():
    sf = slots_from_lineup({"starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1},
                            "flex": 1, "superflex": 1})
    res = free_agent_advice(_fa_board(), _MY, slots=sf)
    ups = {u["player"]: u for u in res["upgrades"]}
    # Now the second QB starts in SUPERFLEX -> full 25, benches nobody. This is the
    # payoff of superflex-aware slots: the same QB was worth only +7 in 1-QB above.
    assert ups["Jalen Hurts"]["gain"] == 25.0 and ups["Jalen Hurts"]["replaces"] is None


def test_free_agent_excludes_rostered_and_own_players_and_ranks_by_gain():
    res = free_agent_advice(_fa_board(), _MY, slots=DEFAULT_SLOTS, exclude=["Nico Collins"])
    names = [u["player"] for u in res["upgrades"]]
    assert "Nico Collins" not in names        # excluded (rostered elsewhere)
    assert all(n not in _MY for n in names)    # never suggests your own players
    gains = [u["gain"] for u in res["upgrades"]]
    assert gains == sorted(gains, reverse=True)   # ranked best-first


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
