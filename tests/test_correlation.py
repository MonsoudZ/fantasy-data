"""Copula correlation: relationship logic, PSD repair, marginal preservation."""

import numpy as np
import pandas as pd
import pytest

from ffdata.correlation import CorrelatedSampler, _rho, _nearest_psd, RHO


def _p(pos, team, opp):
    return pd.Series({"position": pos, "recent_team": team, "opponent_team": opp})


def test_rho_qb_stack_is_the_estimated_value():
    a, b = _p("QB", "KC", "LV"), _p("WR", "KC", "LV")
    assert _rho(a, b, RHO) == RHO["qb_pc"]


def test_rho_is_zero_across_different_games():
    a, b = _p("QB", "KC", "LV"), _p("WR", "SF", "SEA")
    assert _rho(a, b, RHO) == 0.0


def test_rho_bring_back_for_opposing_qb_and_receiver():
    a, b = _p("QB", "KC", "LV"), _p("WR", "LV", "KC")  # same game, opponents
    assert _rho(a, b, RHO) == RHO["opp_qb_pc"]


def test_nearest_psd_is_symmetric_unit_diagonal_and_psd():
    # A non-PSD "correlation" matrix (0.9s can't all coexist).
    S = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    P = _nearest_psd(S)
    assert np.allclose(np.diag(P), 1.0)
    assert np.allclose(P, P.T)
    assert np.linalg.eigvalsh(P).min() >= -1e-9


@pytest.fixture
def sampler():
    rng = np.random.default_rng(0)
    resid = pd.DataFrame({
        "position": ["QB"] * 200 + ["WR"] * 200,
        "pred": np.r_[rng.uniform(10, 25, 200), rng.uniform(5, 20, 200)],
        "residual": np.r_[rng.normal(0, 7, 200), rng.normal(0, 6, 200)],
    })
    return CorrelatedSampler(resid, seed=0)


def test_sample_preserves_the_mean(sampler):
    players = pd.DataFrame({"position": ["QB", "WR"], "recent_team": ["KC", "KC"],
                            "opponent_team": ["LV", "LV"], "pred": [22.0, 14.0]})
    draws = sampler.sample(players, 20000)
    # Correlation must not shift marginals: each row centers on its projection.
    assert draws[0].mean() == pytest.approx(22.0, abs=1.0)
    assert draws[1].mean() == pytest.approx(14.0, abs=1.0)


def test_same_team_qb_receiver_draws_are_positively_correlated(sampler):
    players = pd.DataFrame({"position": ["QB", "WR"], "recent_team": ["KC", "KC"],
                            "opponent_team": ["LV", "LV"], "pred": [22.0, 14.0]})
    draws = sampler.sample(players, 40000)
    assert np.corrcoef(draws[0], draws[1])[0, 1] > 0.1


def test_different_game_players_are_uncorrelated(sampler):
    players = pd.DataFrame({"position": ["QB", "WR"], "recent_team": ["KC", "SF"],
                            "opponent_team": ["LV", "SEA"], "pred": [22.0, 14.0]})
    draws = sampler.sample(players, 40000)
    assert abs(np.corrcoef(draws[0], draws[1])[0, 1]) < 0.05
