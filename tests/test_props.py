"""Props EV/side selection -- picking a wrong side quietly recommends -EV bets."""

import pytest

from ffdata.props import _ev_side, MARKETS


def test_high_model_prob_bets_over_with_positive_ev():
    # Model 80% over, even-money both sides -> over is strongly +EV.
    side, ev = _ev_side(0.80, over_odds=100, under_odds=100)
    assert side == "over"
    assert ev == pytest.approx(0.80 * 1.0 - 0.20)  # 0.60


def test_low_model_prob_bets_under():
    side, ev = _ev_side(0.20, over_odds=100, under_odds=100)
    assert side == "under"
    assert ev == pytest.approx(0.80 * 1.0 - 0.20)  # 0.60


def test_fair_coin_into_vig_is_negative_ev_either_way():
    # 50/50 model into -110/-110 juice: both sides lose ~4.5% -> no edge.
    side, ev = _ev_side(0.50, over_odds=-110, under_odds=-110)
    assert ev < 0


def test_every_market_maps_to_real_positions():
    for market, positions in MARKETS.items():
        assert positions and all(p in {"QB", "RB", "WR", "TE"} for p in positions)
