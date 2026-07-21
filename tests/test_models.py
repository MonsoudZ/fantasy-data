"""Projector blend + residual sampler behavior on synthetic inputs."""

import numpy as np
import pandas as pd
import pytest

from ffdata.projections import TrailingAverageProjector
from ffdata.matchup import ResidualSampler


def test_trailing_average_blends_r3_and_r5():
    proj = TrailingAverageProjector(w3=0.6, w5=0.4)
    df = pd.DataFrame({"fp_r3": [10.0], "fp_r5": [20.0]})
    assert proj.predict(df)[0] == 14.0  # 0.6*10 + 0.4*20


def test_trailing_average_falls_back_when_one_window_missing():
    proj = TrailingAverageProjector(w3=0.6, w5=0.4)
    df = pd.DataFrame({"fp_r3": [np.nan], "fp_r5": [20.0]})
    # r3 fills from r5 -> both 20 -> 20
    assert proj.predict(df)[0] == 20.0


def test_residual_sampler_draws_from_the_matching_bucket():
    # WR residuals; sampler should only ever return pred + one of these values.
    resid = pd.DataFrame({
        "position": ["WR"] * 6,
        "pred": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        "residual": [-1.0, 0.0, 1.0, -3.0, 0.0, 3.0],
    })
    sampler = ResidualSampler(resid, n_bins=2, seed=0)
    draws = sampler.sample("WR", pred=2.0, n=5000)
    offsets = np.unique(np.round(draws - 2.0, 6))
    assert set(offsets).issubset({-1.0, 0.0, 1.0, -3.0, 3.0})


def test_residual_sampler_is_unbiased_on_average():
    resid = pd.DataFrame({
        "position": ["WR"] * 4, "pred": [5.0] * 4,
        "residual": [-2.0, -1.0, 1.0, 2.0],  # mean 0
    })
    sampler = ResidualSampler(resid, n_bins=1, seed=1)
    draws = sampler.sample("WR", pred=5.0, n=20000)
    assert draws.mean() == pytest.approx(5.0, abs=0.1)


def test_residual_sampler_unknown_position_uses_fallback_pool():
    resid = pd.DataFrame({
        "position": ["WR"] * 3, "pred": [1.0, 2.0, 3.0],
        "residual": [0.0, 0.0, 0.0],
    })
    sampler = ResidualSampler(resid, n_bins=1, seed=0)
    draws = sampler.sample("KICKER", pred=7.0, n=10)  # no KICKER bucket
    assert np.allclose(draws, 7.0)
