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

    # Weekly CLI (a Sunday-morning tool):
    python -m ffdata.optimize --week 15 --roster my_players.csv \\
                              --opponent their_players.csv --scoring ppr

    # Or from Python:
    from ffdata.matchup import MatchupSimulator
    from ffdata.optimize import LineupOptimizer
    sim = MatchupSimulator.fit()
    board = sim.project(season=2024, week=15)
    opt = LineupOptimizer(sim).optimize(my_pool, opp_lineup)
    print(opt["optimal_win_prob"], "vs points-lineup", opt["points_win_prob"])

A roster file is one player name per line (a `player` header is fine); names are
matched loosely (case, punctuation, and Jr/Sr/III suffixes are ignored).
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


# --------------------------------------------------------------------------- #
# Weekly CLI
# --------------------------------------------------------------------------- #

import re as _re

_SUFFIX = _re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", _re.I)


def _norm(name: str) -> str:
    """Loose name key for matching a roster CSV to the projection board."""
    n = _SUFFIX.sub("", str(name).lower())
    n = _re.sub(r"[^a-z ]", "", n)
    return _re.sub(r"\s+", " ", n).strip()


def _load_names(path: str) -> list[str]:
    """Read player names: one per line, or the first column of a CSV (header ok)."""
    names = []
    with open(path) as f:
        for i, line in enumerate(f):
            s = line.strip().split(",")[0].strip().strip('"')
            if not s:
                continue
            if i == 0 and s.lower() in ("player", "name", "players", "player_name"):
                continue
            names.append(s)
    return names


def _match(names: list[str], board: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Match roster names to board rows by normalized name."""
    idx = {}
    for _, r in board.iterrows():
        idx.setdefault(_norm(r["player_display_name"]), r)
    rows, missing = [], []
    for name in names:
        r = idx.get(_norm(name))
        (rows.append(r) if r is not None else missing.append(name))
    return (pd.DataFrame(rows).reset_index(drop=True) if rows else board.iloc[:0]), missing


def _print_lineup(title: str, lineup, board: pd.DataFrame) -> None:
    pred = board.set_index("player_display_name")
    print(f"\n{title}")
    total = 0.0
    for slot, name in lineup:
        row = pred.loc[name]
        total += float(row["pred"])
        team = row.get("recent_team", "")
        print(f"  {slot:5} {name:24} {row['position']:3} {team:4} {row['pred']:5.1f}")
    print(f"  {'':5} {'TOTAL':24} {'':3} {'':4} {total:5.1f}")


def main() -> None:
    import argparse
    from .ingest import current_nfl_season
    from .matchup import MatchupSimulator
    from .scoring import PPR, HALF_PPR, STANDARD

    p = argparse.ArgumentParser(
        prog="python -m ffdata.optimize",
        description="Weekly win-probability lineup optimizer.")
    p.add_argument("--week", type=int, required=True, help="NFL week to optimize")
    p.add_argument("--season", type=int, default=current_nfl_season())
    p.add_argument("--roster", required=True,
                   help="your available players: CSV/txt, one name per line")
    p.add_argument("--opponent", help="opponent's players (same format); "
                   "enables win-probability optimization")
    p.add_argument("--scoring", choices=["ppr", "half", "standard"], default="ppr")
    p.add_argument("--projector", choices=["gbm", "neural"], default="gbm",
                   help="gbm is fast; neural is most accurate but slower")
    p.add_argument("--n-sims", type=int, default=20000)
    args = p.parse_args()

    rules = {"ppr": PPR, "half": HALF_PPR, "standard": STANDARD}[args.scoring]
    print(f"Projecting {args.season} week {args.week} "
          f"({args.scoring.upper()}, {args.projector})...", flush=True)
    sim = MatchupSimulator.fit(projector=args.projector, rules=rules)
    board = sim.project(args.season, args.week).reset_index(drop=True)
    if board.empty:
        raise SystemExit(f"No projections for {args.season} week {args.week}. "
                         f"Ingest that season first: python -m ffdata.cli --seasons {args.season}")

    pool, missing = _match(_load_names(args.roster), board)
    if missing:
        print(f"\n[warn] no projection found for: {', '.join(missing)}")
    if pool.empty:
        raise SystemExit("None of your roster names matched the projection board.")

    if args.opponent:
        opp_pool, opp_missing = _match(_load_names(args.opponent), board)
        if opp_missing:
            print(f"[warn] opponent players not found: {', '.join(opp_missing)}")
        opp = _assemble(opp_pool)
        res = LineupOptimizer(sim, n_sims=args.n_sims).optimize(pool, opp)
        print(f"\nOpponent's best lineup projects {res['opp_proj']} pts.")
        _print_lineup(f"RECOMMENDED (max win probability {res['optimal_win_prob']*100:.1f}%, "
                      f"proj {res['optimal_proj']}):", res["optimal_lineup"], board)
        if res["optimal_lineup"] != res["points_lineup"]:
            _print_lineup(f"(for reference) max-points lineup "
                          f"(win {res['points_win_prob']*100:.1f}%, proj {res['points_proj']}):",
                          res["points_lineup"], board)
        else:
            print("\nMax-points lineup is already the highest-win-probability lineup.")
    else:
        lineup = LineupOptimizer(sim)._greedy_points(pool)
        lineup = [(s, n) for s, n, _, _ in lineup]
        _print_lineup("RECOMMENDED lineup (most projected points):", lineup, board)
        print("\nTip: pass --opponent to optimize win probability instead of points.")


if __name__ == "__main__":
    main()
