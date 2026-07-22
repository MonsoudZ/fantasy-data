"""Betting math: American odds, de-vig, and empirical over-probabilities.

Small, dependency-light helpers shared by the odds-facing tools (`props.py`).
Betting math has to be exact -- a wrong de-vig quietly invents fake edges -- so
it lives in one tested place rather than being duplicated per module.

    from ffdata.betting import american_to_prob, american_profit, _prob_over

(These once lived in a game-line edge finder, `edge.py`. That finder measured a
clean negative -- game betting markets are efficient to a public-data model, no
edge survives the vig -- and was pruned; only this reusable math remains.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def american_to_prob(odds: pd.Series | np.ndarray) -> np.ndarray:
    """Convert American odds to their raw (vig-inclusive) implied probability."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # np.where evaluates both branches; the unused one can divide by zero.
        return np.where(odds < 0, -odds / (-odds + 100.0), 100.0 / (odds + 100.0))


def american_profit(odds: float, won: bool) -> float:
    """Profit on a 1-unit stake at American `odds` (push handled by caller)."""
    if not won:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / -odds


def _prob_over(resid: np.ndarray, pred: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Empirical P(outcome > line) = share of residuals clearing (line - pred)."""
    need = (line - pred)[:, None]        # (n, 1)
    return (resid[None, :] > need).mean(axis=1)
