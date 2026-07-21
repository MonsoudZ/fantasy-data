"""Monte Carlo matchup win probability over projections.

Roadmap step 4. A point projection ("Player X: 14.2") can't answer "will my
lineup beat yours?" -- that needs each player's *distribution* of outcomes.

We build one empirically: run the walk-forward backtest, collect the model's
*out-of-sample* residuals (actual - projection), and resample them. Residuals
are position-dependent and heteroscedastic -- a 20-point projection has far more
spread than a 3-point one, and fantasy scoring is right-skewed (a hard floor
near zero, an occasional ceiling game). So residuals are bucketed by position
and projection tier, and a player's simulated week is `projection + a residual
drawn from its own bucket`. No Gaussian assumption.

To score a matchup we sample every player, sum each lineup, and race them over
many draws:

    sim = MatchupSimulator.fit(train_from=2019, resid_seasons=[2023, 2024])
    board = sim.project(season=2024, week=15)          # projections for a week
    a = board.nlargest(7, "pred")                        # two example lineups
    b = board.iloc[7:14]
    print(sim.matchup(a, b))

Limitation (v1): players are sampled independently, so same-game correlation
(a QB and his WR booming together) is not yet modeled -- this slightly
understates the variance of stacked lineups. A game-level shared shock is the
natural next increment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_features
from .projections import GBMProjector, walk_forward, _order_key


class ResidualSampler:
    """Empirical projection residuals, bucketed by position x projection tier."""

    def __init__(self, resid: pd.DataFrame, n_bins: int = 5, seed: int = 0):
        """resid: columns [position, pred, residual] from OOS backtest rows."""
        self.rng = np.random.default_rng(seed)
        self.n_bins = n_bins
        self._edges: dict[str, np.ndarray] = {}
        self._buckets: dict[tuple[str, int], np.ndarray] = {}
        self._fallback = resid["residual"].to_numpy()
        for pos, g in resid.groupby("position"):
            edges = np.quantile(g["pred"], np.linspace(0, 1, n_bins + 1))
            self._edges[pos] = edges
            idx = self._digitize(edges, g["pred"].to_numpy())
            res = g["residual"].to_numpy()
            for b in range(n_bins):
                self._buckets[(pos, b)] = res[idx == b]

    def _digitize(self, edges: np.ndarray, pred) -> np.ndarray:
        # Interior edges only, so bins land in 0..n_bins-1.
        return np.clip(np.digitize(pred, edges[1:-1]), 0, self.n_bins - 1)

    def sample(self, position: str, pred: float, n: int) -> np.ndarray:
        """Draw n simulated point outcomes for one player-week."""
        edges = self._edges.get(position)
        if edges is None:
            pool = self._fallback
        else:
            b = int(self._digitize(edges, np.array([pred]))[0])
            pool = self._buckets.get((position, b))
            if pool is None or len(pool) == 0:
                pool = self._fallback
        return pred + self.rng.choice(pool, size=n, replace=True)


class MatchupSimulator:
    """Projections + a residual sampler = simulated lineup totals and win odds."""

    def __init__(self, feats: pd.DataFrame, projector, sampler: ResidualSampler):
        self._feats = feats.assign(_k=_order_key(feats))
        self.projector = projector
        self.sampler = sampler

    @classmethod
    def fit(
        cls,
        train_from: int = 2019,
        resid_seasons: list[int] | None = None,
        positions: tuple[str, ...] = ("QB", "RB", "WR", "TE"),
        n_bins: int = 5,
        seed: int = 0,
        projector: str = "neural",
    ) -> "MatchupSimulator":
        """Build the simulator. `projector`: "neural" (the promoted GRU, most
        accurate) or "gbm" (faster). Both feed a residual sampler built from
        out-of-sample predictions -- in-sample residuals understate variance."""
        resid_seasons = resid_seasons or [2023, 2024]
        seasons = list(range(train_from, max(resid_seasons) + 1))
        feats = build_features(seasons=seasons, positions=positions)
        if projector == "neural":
            from .neural import NeuralProjector, neural_residuals
            resid = neural_residuals(feats, resid_seasons)
            proj = NeuralProjector()
        else:
            proj = GBMProjector()
            preds = walk_forward(feats, proj, resid_seasons)
            resid = preds.assign(residual=preds["fp"] - preds["pred"])[["position", "pred", "residual"]]
        sampler = ResidualSampler(resid, n_bins, seed)
        return cls(feats, proj, sampler)

    def project(self, season: int, week: int) -> pd.DataFrame:
        """Train on everything before (season, week) and project that week."""
        k = season * 100 + week
        train = self._feats[self._feats["_k"] < k]
        test = self._feats[(self._feats["_k"] == k) & self._feats["fp_r3"].notna()].copy()
        self.projector.fit(train)
        test["pred"] = self.projector.predict(test)
        cols = ["season", "week", "player_display_name", "position", "recent_team", "pred", "fp"]
        return test[[c for c in cols if c in test.columns]].sort_values("pred", ascending=False)

    def simulate_lineup(self, lineup: pd.DataFrame, n_sims: int) -> np.ndarray:
        """Sum sampled outcomes across a lineup's players -> n_sims totals."""
        totals = np.zeros(n_sims)
        for _, row in lineup.iterrows():
            totals += self.sampler.sample(row["position"], row["pred"], n_sims)
        return totals

    def matchup(self, lineup_a: pd.DataFrame, lineup_b: pd.DataFrame, n_sims: int = 20000) -> dict:
        """Head-to-head win probability and total/margin distributions."""
        a = self.simulate_lineup(lineup_a, n_sims)
        b = self.simulate_lineup(lineup_b, n_sims)
        margin = a - b
        return {
            "win_prob_a": round(float((a > b).mean() + 0.5 * (a == b).mean()), 4),
            "proj_a": round(float(lineup_a["pred"].sum()), 1),
            "proj_b": round(float(lineup_b["pred"].sum()), 1),
            "sim_mean_a": round(float(a.mean()), 1),
            "sim_mean_b": round(float(b.mean()), 1),
            "a_p10_p90": (round(float(np.percentile(a, 10)), 1), round(float(np.percentile(a, 90)), 1)),
            "b_p10_p90": (round(float(np.percentile(b, 10)), 1), round(float(np.percentile(b, 90)), 1)),
            "margin_mean": round(float(margin.mean()), 1),
            "margin_std": round(float(margin.std()), 1),
        }


if __name__ == "__main__":
    sim = MatchupSimulator.fit()
    board = sim.project(season=2024, week=15)
    a, b = board.iloc[0:7], board.iloc[7:14]
    print("Lineup A:", list(a["player_display_name"]))
    print("Lineup B:", list(b["player_display_name"]))
    print(sim.matchup(a, b))
