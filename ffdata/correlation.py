"""Same-game correlation for the matchup Monte Carlo.

The independent sampler in matchup.py understates the variance of *stacked*
lineups: a QB and his own receivers boom together (he throws the TD, they catch
it), and everyone in a shootout rides the same game script. Ignoring that makes
a QB-WR stack look safer than it is.

We fix it with a Gaussian copula, which adds correlation while leaving each
player's marginal distribution untouched -- so the ~1pt interval calibration we
validated still holds. Procedure:

  1. Build a correlation matrix over the players being simulated, from their
     relationships (same team QB<->receiver, same game, etc.). The rho values
     are estimated from historical residuals, not guessed (see RHO / estimate).
  2. Draw correlated standard normals, push them through the normal CDF to get
     correlated uniforms, then map each uniform through that player's empirical
     residual quantile -- preserving the marginal exactly.

    from ffdata.correlation import CorrelatedSampler
    cs = CorrelatedSampler(residual_frame)   # same residuals the sampler uses
    totals = cs.sample(lineup_df, n_sims).sum(axis=0)   # correlated lineup total
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

_PC = {"WR", "TE"}

# Residual-correlation by relationship, estimated from 2023-24 (see estimate()).
# The QB<->own-receiver "stack" effect dominates (+0.20, n=8580); same-team
# receivers are ~uncorrelated (target competition cancels the shared game).
RHO = {
    "qb_pc": 0.199, "pc_pc": 0.004, "rb_pc": -0.011, "qb_rb": 0.041,  # same team
    "opp_qb_pc": 0.059, "opp_pc_pc": 0.034,                           # same game, opponents
}


def _rho(a: pd.Series, b: pd.Series, rho: dict) -> float:
    """Correlation between two players from team/opponent/position relationship."""
    if {a["recent_team"], a["opponent_team"]} != {b["recent_team"], b["opponent_team"]}:
        return 0.0  # different games are independent
    pa, pb = a["position"], b["position"]
    both_pc = pa in _PC and pb in _PC
    has = lambda p: p in (pa, pb)
    if a["recent_team"] == b["recent_team"]:                   # same team
        if has("QB") and (pa in _PC or pb in _PC):
            return rho["qb_pc"]
        if both_pc:
            return rho["pc_pc"]
        if has("RB") and (pa in _PC or pb in _PC):
            return rho["rb_pc"]
        if has("QB") and has("RB"):
            return rho["qb_rb"]
        return 0.0
    if has("QB") and (pa in _PC or pb in _PC):                # opponents, bring-back
        return rho["opp_qb_pc"]
    if both_pc:
        return rho["opp_pc_pc"]
    return 0.0


def _nearest_psd(S: np.ndarray) -> np.ndarray:
    """Nearest positive-definite correlation matrix (clip eigenvalues, renormalize)."""
    w, V = np.linalg.eigh((S + S.T) / 2)
    S2 = (V * np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(S2))
    return S2 / np.outer(d, d)


class CorrelatedSampler:
    """Empirical residual buckets + a Gaussian copula over same-game players."""

    def __init__(self, resid: pd.DataFrame, rho: dict | None = None, n_bins: int = 5, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.rho = rho or RHO
        self.n_bins = n_bins
        self._edges: dict[str, np.ndarray] = {}
        self._buckets: dict[tuple[str, int], np.ndarray] = {}
        self._fallback = resid["residual"].to_numpy()
        for pos, g in resid.groupby("position"):
            edges = np.quantile(g["pred"], np.linspace(0, 1, n_bins + 1))
            self._edges[pos] = edges
            idx = np.clip(np.digitize(g["pred"].to_numpy(), edges[1:-1]), 0, n_bins - 1)
            res = g["residual"].to_numpy()
            for b in range(n_bins):
                self._buckets[(pos, b)] = res[idx == b]

    def _pool(self, position: str, pred: float) -> np.ndarray:
        edges = self._edges.get(position)
        if edges is None:
            return self._fallback
        b = int(np.clip(np.digitize([pred], edges[1:-1])[0], 0, self.n_bins - 1))
        pool = self._buckets.get((position, b))
        return pool if pool is not None and len(pool) else self._fallback

    def corr_matrix(self, players: pd.DataFrame) -> np.ndarray:
        rows = players.reset_index(drop=True)
        n = len(rows)
        S = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                S[i, j] = S[j, i] = _rho(rows.iloc[i], rows.iloc[j], self.rho)
        return S

    def sample(self, players: pd.DataFrame, n_sims: int) -> np.ndarray:
        """Correlated outcome matrix (n_players x n_sims); rows align to `players`."""
        rows = players.reset_index(drop=True)
        n = len(rows)
        L = np.linalg.cholesky(_nearest_psd(self.corr_matrix(rows)))
        U = norm.cdf(L @ self.rng.standard_normal((n, n_sims)))     # correlated uniforms
        out = np.empty((n, n_sims))
        for i in range(n):
            pool = self._pool(rows.iloc[i]["position"], rows.iloc[i]["pred"])
            out[i] = rows.iloc[i]["pred"] + np.quantile(pool, U[i], method="linear")
        return out


def estimate(resid: pd.DataFrame) -> dict:
    """Estimate the RHO map from an out-of-sample residual frame.

    `resid` needs columns: player_id, season, week, recent_team, opponent_team,
    position, residual.
    """
    def pair(a, b, keys, same=False):
        m = a.merge(b, on=keys, suffixes=("_i", "_j"))
        m = m[m["player_id_i"] < m["player_id_j"]] if same else m[m["player_id_i"] != m["player_id_j"]]
        return float(np.corrcoef(m["resid_i"], m["resid_j"])[0, 1]) if len(m) >= 30 else 0.0

    r = resid.rename(columns={"residual": "resid"})
    qb, pc, rb = (r[r.position == "QB"], r[r.position.isin(_PC)], r[r.position == "RB"])
    team = ["season", "week", "recent_team"]
    game_i = ["season", "week", "recent_team", "opponent_team"]
    game_j = ["season", "week", "opponent_team", "recent_team"]

    def opp(a, b):
        m = a.merge(b, left_on=game_i, right_on=game_j, suffixes=("_i", "_j"))
        return float(np.corrcoef(m["resid_i"], m["resid_j"])[0, 1]) if len(m) >= 30 else 0.0

    return {
        "qb_pc": round(pair(qb, pc, team), 3),
        "pc_pc": round(pair(pc, pc, team, same=True), 3),
        "rb_pc": round(pair(rb, pc, team), 3),
        "qb_rb": round(pair(qb, rb, team), 3),
        "opp_qb_pc": round(opp(qb, pc), 3),
        "opp_pc_pc": round(opp(pc, pc), 3),
    }
