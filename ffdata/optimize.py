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

Method: sample each candidate player's weekly outcome once, with common random
numbers (every lineup is judged on identical draws, so comparisons are low-
variance), then hill-climb over slot swaps to maximize simulated
P(my_total > opp_total).

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

    def _sample(self, players: pd.DataFrame) -> dict:
        """One outcome vector per player (common random numbers across lineups)."""
        return {r["player_display_name"]: self.sim.sampler.sample(r["position"], r["pred"], self.n_sims)
                for _, r in players.iterrows()}

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

    def optimize(self, pool: pd.DataFrame, opp_lineup: pd.DataFrame) -> dict:
        """Return the max-points lineup vs the win-probability-optimal lineup."""
        vecs = self._sample(pool)
        opp_total = np.sum(list(self._sample(opp_lineup).values()), axis=0)

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

    # Opponent gets the strong top of the board; I get a weaker pool (underdog),
    # where chasing win probability should favor ceiling over safe points.
    opp = _assemble(board.head(40))
    mine = board[~board["player_display_name"].isin(opp["player_display_name"])].iloc[20:120]

    res = LineupOptimizer(sim).optimize(mine, opp)
    print(f"Opponent projected: {res['opp_proj']}\n")
    print(f"Max-points lineup   proj={res['points_proj']}  win%={res['points_win_prob']*100:.1f}")
    print(f"Win-prob-optimal    proj={res['optimal_proj']}  win%={res['optimal_win_prob']*100:.1f}")
    changed = [n for (s, n) in res["optimal_lineup"] if (s, n) not in res["points_lineup"]]
    print(f"\nSwaps the optimizer made vs max-points: {changed}")
