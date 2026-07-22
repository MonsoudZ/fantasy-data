"""Lineup optimizer -- two modes, both beyond "most projected points."

  * Head-to-head (optimize): maximize P(beating a specific opponent). As an
    underdog you want ceiling; as a favorite, floor. A points ranking is blind
    to this; win probability sees it directly.
  * Tournament (optimize_tournament): maximize the lineup's CEILING (a high
    quantile of its own outcome distribution). Large-field contests pay only the
    extreme right tail, so the median is worthless -- you need a top score, which
    means players booming *together*. Ceiling-seeking over correlated draws
    therefore builds stacks (a QB + his receivers), the fat right tail that wins.

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

# A standard fantasy starting lineup. FLEX takes any RB/WR/TE; SUPERFLEX also
# takes a QB (a 2-QB / superflex league, where QBs are far more valuable).
DEFAULT_SLOTS = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX")
_ELIGIBLE = {"QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"},
             "FLEX": {"RB", "WR", "TE"}, "SUPERFLEX": {"QB", "RB", "WR", "TE"}}


def slots_from_lineup(lineup: dict | None) -> tuple:
    """Turn a league lineup config into a slot tuple for the optimizer.

    `lineup` is {"starters": {QB/RB/WR/TE: n}, "flex": n, "superflex": n} -- the
    same shape a Sleeper import produces. Falls back to DEFAULT_SLOTS when the
    config is missing or empty, so a 1-QB league is unaffected and a superflex
    league gets its SUPERFLEX slot (which lets a second QB start).
    """
    if not lineup:
        return DEFAULT_SLOTS
    starters = lineup.get("starters") or {}
    slots = []
    for pos in ("QB", "RB", "WR", "TE"):
        slots += [pos] * int(starters.get(pos, 0) or 0)
    slots += ["FLEX"] * int(lineup.get("flex", 0) or 0)
    slots += ["SUPERFLEX"] * int(lineup.get("superflex", 0) or 0)
    return tuple(slots) if slots else DEFAULT_SLOTS


def _greedy_fill(pool: pd.DataFrame, slots) -> list:
    """Fill each slot with the highest-projected eligible unused player.

    Returns [[slot, name, position, pred], ...]; a slot with no eligible player
    left is simply skipped (a short roster yields a partial lineup).
    """
    used, lineup = set(), []
    ranked = pool.sort_values("pred", ascending=False)
    for slot in slots:
        for _, r in ranked.iterrows():
            name = r["player_display_name"]
            if name not in used and r["position"] in _ELIGIBLE[slot]:
                used.add(name)
                lineup.append([slot, name, r["position"], float(r["pred"])])
                break
    return lineup


class LineupOptimizer:
    def __init__(self, sim, slots=DEFAULT_SLOTS, n_sims: int = 20000):
        self.sim = sim
        self.slots = list(slots)
        self.n_sims = n_sims

    def _draw(self, pool: pd.DataFrame, opp: pd.DataFrame | None, correlated: bool):
        """Joint outcome matrix over pool (+ opponent). Rows align to concat order.

        Correlated (default) draws same-game players together via the copula, so
        stacks carry their real variance and fatter ceilings; falls back to
        independent per-player sampling with no correlated sampler / game context.
        """
        combined = pool if opp is None else pd.concat([pool, opp], ignore_index=True)
        cs = getattr(self.sim, "csampler", None)
        if correlated and cs is not None and "opponent_team" in combined.columns:
            draws = cs.sample(combined, self.n_sims)
        else:
            draws = np.vstack([self.sim.sampler.sample(r["position"], r["pred"], self.n_sims)
                               for _, r in combined.iterrows()])
        return draws, len(pool)

    def _hillclimb(self, pool: pd.DataFrame, score, base: list) -> list:
        """Greedy start `base`, then keep the best slot-swap while it improves `score`."""
        lineup = [row[:] for row in base]
        improved = True
        while improved:
            improved = False
            for i, (slot, _, _, _) in enumerate(lineup):
                names = [x[1] for x in lineup]
                cur = score(names)
                starters = set(names)
                best_gain, best = 1e-9, None
                for _, r in pool.iterrows():
                    q = r["player_display_name"]
                    if q in starters or r["position"] not in _ELIGIBLE[slot]:
                        continue
                    gain = score(names[:i] + [q] + names[i + 1:]) - cur
                    if gain > best_gain:
                        best_gain, best = gain, (q, r["position"], float(r["pred"]))
                if best:
                    lineup[i] = [slot, best[0], best[1], best[2]]
                    improved = True
        return lineup

    def _greedy_points(self, pool: pd.DataFrame) -> list:
        """Fill each slot with the highest-projected eligible unused player."""
        return _greedy_fill(pool, self.slots)

    @staticmethod
    def _winprob(names: list, vecs: dict, opp_total: np.ndarray) -> float:
        my = np.sum([vecs[n] for n in names], axis=0)
        return float((my > opp_total).mean() + 0.5 * (my == opp_total).mean())

    def optimize(self, pool: pd.DataFrame, opp_lineup: pd.DataFrame,
                 correlated: bool = True) -> dict:
        """Head-to-head: max-points lineup vs the win-probability-optimal lineup."""
        pool = pool.reset_index(drop=True)
        draws, n_pool = self._draw(pool, opp_lineup, correlated)
        names = pool["player_display_name"].tolist()
        vecs = {names[i]: draws[i] for i in range(n_pool)}
        opp_total = draws[n_pool:].sum(axis=0)

        score = lambda ns: self._winprob(ns, vecs, opp_total)
        base = self._greedy_points(pool)
        lineup = self._hillclimb(pool, score, base)
        return {
            "opp_proj": round(float(opp_lineup["pred"].sum()), 1),
            "points_lineup": [(s, n) for s, n, _, _ in base],
            "points_proj": round(sum(x[3] for x in base), 1),
            "points_win_prob": round(score([x[1] for x in base]), 4),
            "optimal_lineup": [(s, n) for s, n, _, _ in lineup],
            "optimal_proj": round(sum(x[3] for x in lineup), 1),
            "optimal_win_prob": round(score([x[1] for x in lineup]), 4),
        }

    def optimize_tournament(self, pool: pd.DataFrame, quantile: float = 0.90) -> dict:
        """Large-field mode: maximize the lineup's CEILING (a high quantile of its
        own outcome distribution) instead of beating one opponent.

        Tournaments pay the extreme right tail, so the median doesn't matter --
        you need a top score. Ceiling-seeking over correlated draws is what a
        stacking strategy exploits. Returns the max-points lineup vs the max-
        ceiling lineup, with each one's projection, ceiling, and median.

        FINDING: with our data-measured correlations, pure ceiling-optimization
        stacks only marginally. In an 8-player lineup a QB+WR stack is 2 players,
        so the correlation's variance boost is diluted across the lineup while the
        projection cost of stacking is direct -- creating a stack by dropping ~3
        projected points adds only ~+2.4 ceiling at the 97th pct, a net loss. The
        real-world "always stack in tournaments" heuristic is driven substantially
        by ownership/leverage (differentiating from thousands of entries), which
        needs DFS ownership data nflverse doesn't have. The ceiling objective is
        correct and does trade projection for ceiling; strong stacking would need
        multi-player game-stacks + ownership modeling.
        """
        pool = pool.reset_index(drop=True)
        draws, n_pool = self._draw(pool, None, correlated=True)
        names = pool["player_display_name"].tolist()
        vecs = {names[i]: draws[i] for i in range(n_pool)}
        q = quantile * 100

        def totals(ns):
            return np.sum([vecs[n] for n in ns], axis=0)
        score = lambda ns: float(np.percentile(totals(ns), q))
        base = self._greedy_points(pool)
        lineup = self._hillclimb(pool, score, base)

        def summary(rows):
            ns = [x[1] for x in rows]
            t = totals(ns)
            return {"lineup": [(s, n) for s, n, _, _ in rows],
                    "proj": round(sum(x[3] for x in rows), 1),
                    "ceiling": round(float(np.percentile(t, q)), 1),
                    "median": round(float(np.median(t)), 1),
                    "stacks": _qb_stacks(rows, pool)}
        return {"quantile": quantile, "points": summary(base), "optimal": summary(lineup)}

    def _fill_forced(self, pool: pd.DataFrame, forced: list) -> list | None:
        """Build a valid lineup that MUST include `forced` names; fill the rest by
        projection. Returns None if the forced players can't fit the slots."""
        pos = pool.set_index("player_display_name")["position"].to_dict()
        pred = pool.set_index("player_display_name")["pred"].to_dict()
        slots = list(self.slots)
        assign, used = [None] * len(slots), set()
        for name in forced:  # place forced players first (natural slot, then FLEX)
            p = pos.get(name)
            if p is None or name in used:
                return None
            spots = ([i for i, s in enumerate(slots) if s == p and assign[i] is None]
                     + [i for i, s in enumerate(slots)
                        if s == "FLEX" and assign[i] is None and p in _ELIGIBLE["FLEX"]])
            if not spots:
                return None
            assign[spots[0]] = name
            used.add(name)
        for i, s in enumerate(slots):  # fill the remaining slots greedily by pred
            if assign[i] is not None:
                continue
            for _, r in pool.sort_values("pred", ascending=False).iterrows():
                nm = r["player_display_name"]
                if nm not in used and r["position"] in _ELIGIBLE[s]:
                    assign[i] = nm
                    used.add(nm)
                    break
            if assign[i] is None:
                return None
        return [[slots[i], assign[i], pos[assign[i]], float(pred[assign[i]])] for i in range(len(slots))]

    def optimize_game_stack(self, pool: pd.DataFrame, quantile: float = 0.95,
                            stack_size: int = 2, bringback: int = 1, max_qbs: int = 8) -> dict:
        """Best CEILING lineup built AROUND a game stack -- how real DFS tools work.

        A game stack concentrates correlated players (a QB + `stack_size` of his
        own receivers + `bringback` receivers from the opponent) instead of one
        diluted pair, so multiple positive correlations compound into a genuinely
        fatter right tail. We enumerate each candidate QB's stack, fill the rest
        of the lineup to maximize the ceiling, and keep the best; the max-points
        lineup is returned alongside for the tradeoff.

        FINDING: this builds a real stack (QB + own receivers + bring-back), but
        at data-measured correlations even a concentrated 4-player game stack has
        a LOWER ceiling than the unconstrained max-points lineup -- forcing a
        non-optimal QB + a lesser receiver costs ~10 projected pts, which the
        correlation boost can't recover even at the 95th pct. Consistent across
        pure-ceiling, elite-pair, and game-stack tests: stacking doesn't win on
        raw ceiling with our correlations. Its real value is *leverage* (being
        different from the field), which needs DFS ownership data. Use this mode
        to impose a stack as a deliberate leverage play and get the best lineup
        within it -- exactly how DFS optimizers apply stack rules.
        """
        pool = pool.reset_index(drop=True)
        draws, n_pool = self._draw(pool, None, correlated=True)
        names = pool["player_display_name"].tolist()
        vecs = {names[i]: draws[i] for i in range(n_pool)}
        q = quantile * 100
        ceiling = lambda ns: float(np.percentile(np.sum([vecs[x] for x in ns], axis=0), q))

        best = None
        for _, qb in pool[pool["position"] == "QB"].sort_values("pred", ascending=False).head(max_qbs).iterrows():
            mates = pool[(pool["recent_team"] == qb["recent_team"])
                         & pool["position"].isin(["WR", "TE"])].sort_values("pred", ascending=False)
            opps = pool[(pool["recent_team"] == qb["opponent_team"])
                        & pool["position"].isin(["WR", "TE"])].sort_values("pred", ascending=False)
            if len(mates) < stack_size or len(opps) < bringback:
                continue
            forced = ([qb["player_display_name"]] + list(mates.head(stack_size)["player_display_name"])
                      + list(opps.head(bringback)["player_display_name"]))
            lineup = self._fill_forced(pool, forced)
            if lineup is None:
                continue
            c = ceiling([x[1] for x in lineup])
            if best is None or c > best[0]:
                best = (c, lineup, qb["player_display_name"], qb["recent_team"], forced)

        def summary(rows):
            t = np.sum([vecs[x[1]] for x in rows], axis=0)
            return {"lineup": [(s, n) for s, n, _, _ in rows],
                    "proj": round(sum(x[3] for x in rows), 1),
                    "ceiling": round(float(np.percentile(t, q)), 1),
                    "median": round(float(np.median(t)), 1),
                    "stacks": _qb_stacks(rows, pool)}
        pts = summary(self._greedy_points(pool))
        if best is None:
            return {"quantile": quantile, "stack": None, "points": pts, "optimal": pts}
        _, lineup, qb_name, team, forced = best
        out = summary(lineup)
        out["stack"] = {"qb": qb_name, "team": team, "members": forced}
        return {"quantile": quantile, "points": pts, "optimal": out}


def _qb_stacks(lineup: list, pool: pd.DataFrame) -> list:
    """Teams in the lineup fielding a QB plus at least one of his pass-catchers."""
    team = pool.set_index("player_display_name")["recent_team"] if "recent_team" in pool else {}
    pos = pool.set_index("player_display_name")["position"]
    by_team = {}
    for row in lineup:
        name = row[1]
        t = team.get(name) if hasattr(team, "get") else None
        if t is not None:
            by_team.setdefault(t, []).append(pos.get(name))
    return [t for t, ps in by_team.items() if "QB" in ps and any(p in ("WR", "TE") for p in ps)]


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


def free_agent_advice(board: pd.DataFrame, roster: list[str], slots=DEFAULT_SLOTS,
                      exclude: list[str] | None = None, top: int = 15,
                      pool_cap: int = 200) -> dict:
    """Rank available players by how much they'd add to your *starting* lineup.

    Waiver value is not raw projection -- a highly projected WR does nothing for
    you if your WRs already outproject him. So for each free agent we recompute
    the projection-optimal starting lineup with him added and measure the
    MARGINAL gain over your current best; a positive gain means he cracks your
    lineup, and we name the starter he'd bench. This is deliberately projection-
    based (expected starting points added), the honest metric for a season-long
    pickup -- not the Monte Carlo win-prob objective, which answers a different
    question (winning one specific matchup).

    board: a week's projection board (player_display_name/position/pred [/recent_team]).
    roster: your players (loosely name-matched). exclude: names to treat as
    unavailable (e.g. rostered by other managers). Returns your baseline starting
    projection plus the ranked upgrades.
    """
    b = board.reset_index(drop=True)
    mine, _ = _match(roster or [], b)
    base = _greedy_fill(mine, slots)
    base_proj = round(sum(x[3] for x in base), 1)
    base_names = {x[1] for x in base}

    taken = {_norm(n) for n in (roster or [])} | {_norm(n) for n in (exclude or [])}
    free = b[~b["player_display_name"].map(lambda s: _norm(s) in taken)]
    free = free.sort_values("pred", ascending=False).head(pool_cap)

    out = []
    for _, fa in free.iterrows():
        aug = pd.concat([mine, fa.to_frame().T], ignore_index=True) if len(mine) else fa.to_frame().T
        new = _greedy_fill(aug, slots)
        gain = sum(x[3] for x in new) - sum(x[3] for x in base)
        if gain <= 1e-6:
            continue
        dropped = base_names - {x[1] for x in new}     # the starter he displaces
        out.append({"player": fa["player_display_name"], "position": fa["position"],
                    "team": str(fa.get("recent_team", "")), "proj": round(float(fa["pred"]), 1),
                    "gain": round(float(gain), 1),
                    "replaces": next(iter(dropped)) if dropped else None})
    out.sort(key=lambda d: d["gain"], reverse=True)
    return {"starter_proj": base_proj, "starters": len(base), "upgrades": out[:top]}


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
        description="Weekly lineup optimizer (head-to-head win prob or tournament ceiling).")
    p.add_argument("--week", type=int, required=True, help="NFL week to optimize")
    p.add_argument("--season", type=int, default=current_nfl_season())
    p.add_argument("--roster", required=True,
                   help="your available players: CSV/txt, one name per line")
    p.add_argument("--mode", choices=["h2h", "tournament", "stack"], default="h2h",
                   help="h2h: beat one opponent; tournament: max ceiling; "
                   "stack: best ceiling built around a QB game stack")
    p.add_argument("--opponent", help="opponent's players (h2h mode); "
                   "enables win-probability optimization")
    p.add_argument("--ceiling", type=float, default=0.90,
                   help="tournament/stack mode: which quantile to maximize")
    p.add_argument("--stack-size", type=int, default=2,
                   help="stack mode: same-team receivers to pair with the QB")
    p.add_argument("--bringback", type=int, default=1,
                   help="stack mode: opponent receivers to bring back")
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

    opt = LineupOptimizer(sim, n_sims=args.n_sims)
    if args.mode == "stack":
        res = opt.optimize_game_stack(pool, quantile=args.ceiling,
                                      stack_size=args.stack_size, bringback=args.bringback)
        best, pts = res["optimal"], res["points"]
        pct = int(args.ceiling * 100)
        if best.get("stack"):
            s = best["stack"]
            print(f"\nGame stack: {s['team']} {s['qb']} + {', '.join(s['members'][1:])}")
        _print_lineup(f"STACK lineup (max {pct}th-pct ceiling {best['ceiling']}, "
                      f"proj {best['proj']}, median {best['median']}):", best["lineup"], board)
        print(f"\nvs max-points lineup: ceiling {pts['ceiling']}, proj {pts['proj']} "
              f"-- the stack trades {round(pts['proj']-best['proj'],1)} proj for "
              f"{round(best['ceiling']-pts['ceiling'],1):+} ceiling.")
    elif args.mode == "tournament":
        res = opt.optimize_tournament(pool, quantile=args.ceiling)
        pts, best = res["points"], res["optimal"]
        pct = int(args.ceiling * 100)
        _print_lineup(f"TOURNAMENT lineup (max {pct}th-pct ceiling {best['ceiling']}, "
                      f"proj {best['proj']}, median {best['median']}):", best["lineup"], board)
        print(f"\nQB stacks: {best['stacks'] or 'none'}")
        if best["lineup"] != pts["lineup"]:
            _print_lineup(f"(for reference) max-points lineup "
                          f"(ceiling {pts['ceiling']}, proj {pts['proj']}):", pts["lineup"], board)
            print(f"\nThe ceiling lineup trades {round(pts['proj']-best['proj'],1)} projected pts "
                  f"for +{round(best['ceiling']-pts['ceiling'],1)} ceiling -- the tournament tradeoff.")
        else:
            print("\nMax-points lineup already has the highest ceiling.")
    elif args.opponent:
        opp_pool, opp_missing = _match(_load_names(args.opponent), board)
        if opp_missing:
            print(f"[warn] opponent players not found: {', '.join(opp_missing)}")
        opp = _assemble(opp_pool)
        res = opt.optimize(pool, opp)
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
        lineup = [(s, n) for s, n, _, _ in opt._greedy_points(pool)]
        _print_lineup("RECOMMENDED lineup (most projected points):", lineup, board)
        print("\nTip: --opponent optimizes win probability; --mode tournament maximizes ceiling.")


if __name__ == "__main__":
    main()
