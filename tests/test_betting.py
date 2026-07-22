"""Betting math has to be exact -- a wrong de-vig quietly invents fake edges."""

import numpy as np
import pytest

from ffdata.betting import american_to_prob, american_profit, _prob_over


def test_american_to_prob_known_values():
    p = american_to_prob([-110, 100, -200, 150])
    assert p[0] == pytest.approx(110 / 210)   # -110 -> 0.5238
    assert p[1] == pytest.approx(0.5)         # +100 -> even
    assert p[2] == pytest.approx(200 / 300)   # -200 -> 0.6667
    assert p[3] == pytest.approx(100 / 250)   # +150 -> 0.40


def test_devig_two_sided_market_normalizes_to_one():
    over, under = american_to_prob([-110]), american_to_prob([-110])
    fair = over / (over + under)
    assert fair[0] == pytest.approx(0.5)      # symmetric juice -> 50/50 fair


def test_american_profit_payouts():
    assert american_profit(100, won=True) == pytest.approx(1.0)
    assert american_profit(150, won=True) == pytest.approx(1.5)
    assert american_profit(-110, won=True) == pytest.approx(100 / 110)
    assert american_profit(-110, won=False) == -1.0
    assert american_profit(200, won=False) == -1.0


def test_prob_over_is_the_empirical_residual_tail():
    resid = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    # pred 40, line 41 -> need residual > 1 -> {2, 5} -> 2/5 = 0.4
    p = _prob_over(resid, np.array([40.0]), np.array([41.0]))
    assert p[0] == pytest.approx(0.4)


def test_prob_over_monotonic_in_prediction():
    resid = np.linspace(-10, 10, 101)
    line = np.array([45.0, 45.0])
    p = _prob_over(resid, np.array([40.0, 50.0]), line)
    assert p[1] > p[0]  # a higher projected total -> higher P(over)
