"""Win-probability lineup optimizer.

Picks the roster that maximizes the probability of BEATING a specific opponent
-- not the roster with the most projected points. Those two differ, and that
gap is the whole point: as an underdog you want ceiling (variance) to have any
shot; as a favorite you want floor (safety) to protect a lead. A points ranking
is blind to this; win probability sees it directly.

We proved (matchup.py) that our Monte Carlo intervals are calibrated to ~1pt
out of sample, so the win-probability objective rests on honest uncertainty.
This is where the projection work pays off despite the irreducible accuracy
floor: you can't out-predict the noise, but you can make better *decisions*
under it.

Method: draw the whole candidate pool (and the opponent) ONCE, jointly, through
the correlated sampler (correlation.py) -- so a QB and his own receivers share
their real +0.20 correlation, and every lineup is judged on identical draws
(common random numbers, low-variance comparisons). Then hill-climb over slot
swaps to maximize simulated P(my_total > opp_total). Because the draw is
correlated, the optimizer now sees a stack's true ceiling and will build one
when chasing win probability as an underdog -- the whole point of stacking.

    from ffdata.matchup import MatchupSimulator
    from ffdata.optimize import LineupOptimizer
    sim = MatchupSimulator.fit()
    board = sim.project(season=2024, week=15)
    opt = LineupOptimizer(sim).optimize(my_pool, opp_lineup)
    print(opt["optimal_win_prob"], "vs points-lineup", opt["points_win_prob"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A standard fantasy starting lineup. FLEX takes any RB/WR/TE.
DEFAULT_SLOTS = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX")
_ELIGIBLE = {"QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"},
             "FLEX": {"RB", "WR", "TE"}}


class LineupOptimizer:
    def __init__(self, sim, slots=DEFAULT_SLOTS, n_sims: int = 20000):
        self.sim = sim
        self.slots = list(slots)
        self.n_sims = n_sims

    def _draw(self, pool: pd.DataFrame, opp: pd.DataFrame, correlated: bool):
        """Joint outcome matrix over pool + opponent. Rows align to concat order.

        Correlated (default) draws same-game players together via the copula, so
        stacks carry their real variance; falls back to independent per-player
        sampling if no correlated sampler or game context is available.
        """
        combined = pd.concat([pool, opp], ignore_index=True)
        cs = getattr(self.sim, "csampler", None)
        if correlated and cs is not None and "opponent_team" in combined.columns:
            draws = cs.sample(combined, self.n_sims)
        else:
            draws = np.vstack([self.sim.sampler.sample(r["position"], r["pred"], self.n_sims)
                               for _, r in combined.iterrows()])
        return draws, len(pool)

    def _greedy_points(self, pool: pd.DataFrame) -> list:
        """Fill each slot with the highest-projected eligible unused player."""
        used, lineup = set(), []
        ranked = pool.sort_values("pred", ascending=False)
        for slot in self.slots:
            for _, r in ranked.iterrows():
                name = r["player_display_name"]
                if name not in used and r["position"] in _ELIGIBLE[slot]:
                    used.add(name)
                    lineup.append([slot, name, r["position"], float(r["pred"])])
                    break
        return lineup

    @staticmethod
    def _winprob(names: list, vecs: dict, opp_total: np.ndarray) -> float:
        my = np.sum([vecs[n] for n in names], axis=0)
        return float((my > opp_total).mean() + 0.5 * (my == opp_total).mean())

    def optimize(self, pool: pd.DataFrame, opp_lineup: pd.DataFrame,
                 correlated: bool = True) -> dict:
        """Return the max-points lineup vs the win-probability-optimal lineup."""
        pool = pool.reset_index(drop=True)
        draws, n_pool = self._draw(pool, opp_lineup, correlated)
        names = pool["player_display_name"].tolist()
        vecs = {names[i]: draws[i] for i in range(n_pool)}
        opp_total = draws[n_pool:].sum(axis=0)

        base = self._greedy_points(pool)
        base_names = [x[1] for x in base]
        base_wp = self._winprob(base_names, vecs, opp_total)

        lineup = [row[:] for row in base]
        improved = True
        while improved:
            improved = False
            for i, (slot, _, _, _) in enumerate(lineup):
                names = [x[1] for x in lineup]
                cur_wp = self._winprob(names, vecs, opp_total)
                starters = set(names)
                best_gain, best = 1e-9, None
                for _, r in pool.iterrows():
                    q = r["player_display_name"]
                    if q in starters or r["position"] not in _ELIGIBLE[slot]:
                        continue
                    trial = names[:i] + [q] + names[i + 1:]
                    gain = self._winprob(trial, vecs, opp_total) - cur_wp
                    if gain > best_gain:
                        best_gain, best = gain, (q, r["position"], float(r["pred"]))
                if best:
                    lineup[i] = [slot, best[0], best[1], best[2]]
                    improved = True

        opt_names = [x[1] for x in lineup]
        return {
            "opp_proj": round(float(opp_lineup["pred"].sum()), 1),
            "points_lineup": [(s, n) for s, n, _, _ in base],
            "points_proj": round(sum(x[3] for x in base), 1),
            "points_win_prob": round(base_wp, 4),
            "optimal_lineup": [(s, n) for s, n, _, _ in lineup],
            "optimal_proj": round(sum(x[3] for x in lineup), 1),
            "optimal_win_prob": round(self._winprob(opt_names, vecs, opp_total), 4),
        }


def _assemble(board: pd.DataFrame, slots=DEFAULT_SLOTS) -> pd.DataFrame:
    """Greedily pull a valid starting lineup off the top of a projection board."""
    used, rows = set(), []
    for slot in slots:
        for _, r in board.iterrows():
            if r["player_display_name"] not in used and r["position"] in _ELIGIBLE[slot]:
                used.add(r["player_display_name"])
                rows.append(r)
                break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .matchup import MatchupSimulator

    sim = MatchupSimulator.fit()
    board = sim.project(season=2024, week=15).reset_index(drop=True)

    # Opponent gets the strong top of the board (I'm the underdog). Build my pool
    # so a QB stack is actually available: the top QB plus his own receivers, then
    # a spread of mid-tier players to fill the rest.
    opp = _assemble(board.head(28))
    rest = board[~board["player_display_name"].isin(opp["player_display_name"])].reset_index(drop=True)
    qb = rest[rest["position"] == "QB"].iloc[0]
    stack = rest[(rest["recent_team"] == qb["recent_team"]) & rest["position"].isin(["QB", "WR", "TE"])]
    mine = pd.concat([stack, rest.iloc[10:90]]).drop_duplicates("player_display_name")

    opt = LineupOptimizer(sim)
    corr = opt.optimize(mine, opp, correlated=True)
    indep = opt.optimize(mine, opp, correlated=False)

    def stack_teams(lineup):
        pos = mine.set_index("player_display_name")
        teams = {}
        for _, n in lineup:
            r = pos.loc[n]
            teams.setdefault(r["recent_team"], []).append(r["position"])
        return [t for t, ps in teams.items() if "QB" in ps and any(p in ("WR", "TE") for p in ps)]

    print(f"Opponent projected: {corr['opp_proj']}\n")
    print(f"Max-points lineup       proj={corr['points_proj']}  win%={corr['points_win_prob']*100:.1f}")
    print(f"Optimal (independent)   proj={indep['optimal_proj']}  win%={indep['optimal_win_prob']*100:.1f}")
    print(f"Optimal (correlated)    proj={corr['optimal_proj']}  win%={corr['optimal_win_prob']*100:.1f}")
    print(f"\nQB stacks built -- independent: {stack_teams(indep['optimal_lineup'])}"
          f" | correlated: {stack_teams(corr['optimal_lineup'])}")
