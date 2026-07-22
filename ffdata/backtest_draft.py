"""Retrospective draft-and-win backtest: draft blind, replay the real season.

The honest test of the whole decision stack. Draft a team using ONLY preseason
information (our leak-free `draft_board`), then replay the season's ACTUAL weekly
results -- setting each week's lineup the same way the app advises it -- to see
where the team would have finished and how often it would have won the title.
Opponents draft off a naive baseline (last year's points), so the gap isolates
what our value model actually adds over "just take last year's studs."

Leak-free by construction: the draft sees `draft_board(season)` (prior-year
features + preseason-known context only); the replay sees the real weekly points;
the two never meet until the draft is done. Randomizing the draft slot (and the
schedule) over many sims turns one fixed real season into a distribution of
finishes -- a championship *rate*, not a single lucky/unlucky run.

    from ffdata.backtest_draft import run_backtest
    run_backtest(2024, sims=200)     # our title rate vs a naive-drafting league

⚠️  Needs the data lake (weekly + the prior season). The pure simulation engine
below -- snake draft, lineup replay, schedule, standings, playoffs -- is unit-
tested on synthetic frames. `run_backtest` itself needs ingested data and hasn't
been run against a real season in this environment (no lake / no egress); the
numbers it prints are only as trustworthy as the projections feeding it (see the
draft-board and K/DST honesty notes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .optimize import _ELIGIBLE, _norm, slots_from_lineup
from .scoring import PPR, ScoringRules

# A draft with just the skill positions the board ranks (K/DEF are streamed, not
# drafted -- see the draft-board scope note). Bench depth beyond the starters.
DRAFT_SLOTS = ("QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX")
DEFAULT_LIMITS = {"QB": 3, "RB": 8, "WR": 8, "TE": 3}   # positional roster caps


# --------------------------------------------------------------------------- #
# Snake draft
# --------------------------------------------------------------------------- #

def snake_order(n_teams: int, rounds: int) -> list[int]:
    """Team index to pick at each overall selection, snaking each round."""
    order: list[int] = []
    for r in range(rounds):
        order.extend(range(n_teams) if r % 2 == 0 else range(n_teams - 1, -1, -1))
    return order


def run_snake_draft(team_boards: list[list[dict]], rounds: int,
                    limits: dict | None = None) -> list[list[dict]]:
    """Snake-draft `rounds` players per team.

    team_boards[t] is team t's ranked list of player dicts (each with `player`
    and `position`); a team always takes the highest player still available that
    fits an open positional slot. `limits` caps roster count per position so a
    team can't hoard one position. Returns a roster (list of player dicts) per team.
    """
    n = len(team_boards)
    limits = limits or DEFAULT_LIMITS
    taken: set[str] = set()
    counts = [dict() for _ in range(n)]
    rosters: list[list[dict]] = [[] for _ in range(n)]
    for team in snake_order(n, rounds):
        for p in team_boards[team]:
            name, pos = p["player"], p["position"]
            if _norm(name) in taken or counts[team].get(pos, 0) >= limits.get(pos, 99):
                continue
            taken.add(_norm(name))
            counts[team][pos] = counts[team].get(pos, 0) + 1
            rosters[team].append(p)
            break
    return rosters


# --------------------------------------------------------------------------- #
# Season replay (lineups set exactly the way the app advises them)
# --------------------------------------------------------------------------- #

def best_week_total(roster: list[dict], week_points: dict, slots=DRAFT_SLOTS) -> float:
    """Best legal starting total for one week, given that week's actual points.

    The same greedy slot-fill the optimizer uses (highest-scoring eligible player
    per slot), so the replay grades a team exactly the way the lineup tab would
    set it. Kept as a pure loop (no DataFrame) because it's the backtest hot path
    -- one call per team per week per sim.
    """
    players = sorted(((p["position"], float(week_points.get(_norm(p["player"]), 0.0)))
                      for p in roster), key=lambda x: x[1], reverse=True)
    used = [False] * len(players)
    total = 0.0
    for slot in slots:
        elig = _ELIGIBLE[slot]
        for i, (pos, pts) in enumerate(players):
            if not used[i] and pos in elig:
                used[i] = True
                total += pts
                break
    return round(total, 2)


def replay(rosters: list[list[dict]], weekly: pd.DataFrame, slots=DRAFT_SLOTS) -> np.ndarray:
    """Score every team every week off the actual results. `weekly` has columns
    player/week/fp. Returns an (n_teams x n_weeks) points matrix."""
    weeks = sorted(weekly["week"].unique())
    by_week = {w: dict(zip(weekly.loc[weekly.week == w, "player"].map(_norm),
                           weekly.loc[weekly.week == w, "fp"])) for w in weeks}
    return np.array([[best_week_total(r, by_week[w], slots) for w in weeks] for r in rosters])


# --------------------------------------------------------------------------- #
# Schedule, standings, playoffs
# --------------------------------------------------------------------------- #

def round_robin(n_teams: int, weeks: int) -> list[list[tuple[int, int]]]:
    """Circle-method pairings for each week (repeats the rotation past one cycle)."""
    teams = list(range(n_teams))
    if n_teams % 2:
        teams.append(-1)  # bye marker
    m = len(teams)
    sched = []
    for w in range(weeks):
        rot = [teams[0]] + teams[1:][-(w % (m - 1)):] + teams[1:][:-(w % (m - 1))] if w % (m - 1) else teams[:]
        pairs = [(rot[i], rot[m - 1 - i]) for i in range(m // 2) if -1 not in (rot[i], rot[m - 1 - i])]
        sched.append(pairs)
    return sched


def standings(scores: np.ndarray, sched: list[list[tuple[int, int]]]) -> list[dict]:
    """Head-to-head W/L over the scheduled weeks, points-for as the tiebreak.
    Returns teams sorted best-first: [{team, wins, losses, pf}, ...]."""
    n = scores.shape[0]
    wins = np.zeros(n)
    pf = scores[:, :len(sched)].sum(axis=1)
    for w, pairs in enumerate(sched):
        for a, b in pairs:
            if scores[a, w] > scores[b, w]:
                wins[a] += 1
            elif scores[b, w] > scores[a, w]:
                wins[b] += 1
            else:
                wins[a] += 0.5
                wins[b] += 0.5
    table = [{"team": t, "wins": float(wins[t]), "pf": float(pf[t])} for t in range(n)]
    table.sort(key=lambda r: (r["wins"], r["pf"]), reverse=True)
    return table


def playoffs(seeds: list[int], scores: np.ndarray, playoff_weeks: list[int]) -> int:
    """Single-elimination bracket (higher seed advances on ties). `seeds` is the
    playoff field best-first; one playoff week per round. Returns the champion."""
    field = list(seeds)
    for wk in playoff_weeks:
        if len(field) == 1:
            break
        # Re-seed each round: 1 vs last, 2 vs second-last, ...
        nxt = []
        for i in range(len(field) // 2):
            a, b = field[i], field[len(field) - 1 - i]
            nxt.append(a if scores[a, wk] >= scores[b, wk] else b)
        if len(field) % 2:                    # odd field: top seed byes
            nxt.insert(0, field[len(field) // 2])
        field = _reseed(nxt, seeds)
    return field[0]


def _reseed(winners: list[int], original_seeds: list[int]) -> list[int]:
    """Order survivors by their original seed (best seed first)."""
    rank = {t: i for i, t in enumerate(original_seeds)}
    return sorted(winners, key=lambda t: rank[t])


# --------------------------------------------------------------------------- #
# One full season from a set of rosters
# --------------------------------------------------------------------------- #

def simulate_season(rosters: list[list[dict]], weekly: pd.DataFrame, slots=DRAFT_SLOTS,
                    reg_weeks: int = 14, playoff_teams: int = 6,
                    playoff_weeks: tuple[int, ...] = (15, 16, 17)) -> dict:
    """Replay -> regular-season standings -> playoffs -> champion. Weeks are the
    matrix column index+1; only weeks that exist in `weekly` are used."""
    scores = replay(rosters, weekly, slots)
    n_weeks = scores.shape[1]
    reg = min(reg_weeks, n_weeks)
    sched = round_robin(len(rosters), reg)
    table = standings(scores, sched)
    seeds = [r["team"] for r in table[:playoff_teams]]
    pw = [w - 1 for w in playoff_weeks if w - 1 < n_weeks]     # to column indices
    champ = playoffs(seeds, scores, pw) if pw and len(seeds) > 1 else (seeds[0] if seeds else 0)
    return {"standings": table, "seeds": seeds, "champion": champ, "scores": scores}


# --------------------------------------------------------------------------- #
# Orchestration over the real lake (leak-free): draft our board, replay actuals
# --------------------------------------------------------------------------- #

def _actual_weekly(con, season: int, rules: ScoringRules) -> pd.DataFrame:
    """Actual per-player weekly fantasy points for the season (the replay truth)."""
    from .scoring import score
    weekly = con.sql(f"""
        select * from weekly
        where season = {season} and season_type = 'REG'
              and position in ('QB','RB','WR','TE')
    """).df()
    scored = score(weekly, rules, col="fp")
    return (scored.groupby(["player_display_name", "week"], as_index=False)
            .agg(position=("position", "first"), fp=("fp", "sum"))
            .rename(columns={"player_display_name": "player"}))


def _naive_board(con, season: int, rules: ScoringRules) -> list[dict]:
    """The baseline draft board: last season's actual points, best-first. This is
    the 'just take last year's studs' strategy our value model has to beat."""
    from .draft import _season_agg
    agg = _season_agg(con, rules)
    prior = agg[agg["season"] == season - 1].sort_values("fp", ascending=False)
    return [{"player": r["player"], "position": r["position"]} for _, r in prior.iterrows()]


def _board_list(board: pd.DataFrame) -> list[dict]:
    """draft_board DataFrame -> the ranked player-dict list the draft consumes."""
    return [{"player": r["player"], "position": r["position"]} for _, r in board.iterrows()]


def _finish(champ_seeds, team: int) -> int:
    """1-based finishing place of `team` given the final standings order."""
    return next((i + 1 for i, r in enumerate(champ_seeds) if r["team"] == team), len(champ_seeds))


def run_backtest(season: int, rules: ScoringRules = PPR, league: dict | None = None,
                 sims: int = 200, teams: int = 12, rounds: int = 14, con=None,
                 seed: int = 0) -> dict:
    """Draft-and-win backtest for one real season (needs the lake).

    For `sims` random draft slots: draft OUR team from `draft_board(season)` while
    the other `teams-1` managers draft off the naive last-year board, replay the
    real weekly results, and record our finish + whether we won it all. For each
    sim we also run a control where OUR slot drafts naively too -- so the deltas
    (title rate, playoff rate, mean finish) isolate what our board adds over the
    baseline, on identical schedules.
    """
    from .draft import draft_board
    if con is None:
        from .db import connect
        con = connect()

    our = _board_list(draft_board(season, league, rules=rules, con=con))
    naive = _naive_board(con, season, rules)
    weekly = _actual_weekly(con, season, rules)
    slots = slots_from_lineup(league) if league else DRAFT_SLOTS
    slots = tuple(s for s in slots if s in ("QB", "RB", "WR", "TE", "FLEX", "SUPERFLEX"))
    rng = np.random.default_rng(seed)

    agg = {"ours": {"titles": 0, "playoffs": 0, "finish": [], "points": []},
           "naive": {"titles": 0, "playoffs": 0, "finish": [], "points": []}}
    for _ in range(sims):
        slot = int(rng.integers(0, teams))
        for label, my_board in (("ours", our), ("naive", naive)):
            boards = [my_board if t == slot else naive for t in range(teams)]
            rosters = run_snake_draft(boards, rounds)
            res = simulate_season(rosters, weekly, slots)
            a = agg[label]
            a["titles"] += int(res["champion"] == slot)
            a["playoffs"] += int(slot in res["seeds"])
            a["finish"].append(_finish(res["standings"], slot))
            a["points"].append(float(res["scores"][slot, :14].sum()))

    def summarize(a):
        return {"title_rate": round(a["titles"] / sims, 3),
                "playoff_rate": round(a["playoffs"] / sims, 3),
                "mean_finish": round(float(np.mean(a["finish"])), 2),
                "mean_points": round(float(np.mean(a["points"])), 1)}

    ours, base = summarize(agg["ours"]), summarize(agg["naive"])
    return {"season": season, "sims": sims, "teams": teams,
            "ours": ours, "naive": base,
            "title_lift": round(ours["title_rate"] - base["title_rate"], 3),
            "playoff_lift": round(ours["playoff_rate"] - base["playoff_rate"], 3),
            "fair_title_rate": round(1 / teams, 3)}


def prop_accuracy(season: int, markets: list[str] | None = None, con=None) -> pd.DataFrame:
    """How honest our prop guesses are: per-market projection MAE and P(over)
    interval calibration, out of sample (pool from `season-1`, evaluate `season`).

    Reuses the prop engine's own residual/projection path -- so this is the same
    P(over) that `price_props` turns into edges. Calibration near nominal (cover80
    ~0.80) means the probabilities driving prop picks are trustworthy; MAE is the
    raw point accuracy. (No market odds ship with nflverse, so hit-rate-vs-book
    can't be computed here -- calibration is the honest stand-in.)
    """
    from .features import build_features
    from .ingest import FIRST_SEASON
    from .props import MARKETS, _oos_residuals, calibrate
    markets = markets or list(MARKETS)
    if con is not None:
        feats = build_features(seasons=list(range(FIRST_SEASON, season + 1)), con=con)
    else:
        feats = build_features(seasons=list(range(FIRST_SEASON, season + 1)))
    rows = []
    for market in markets:
        if market not in MARKETS:
            continue
        ev = _oos_residuals(feats, market, MARKETS[market], season)
        mae = float((ev["pred"] - ev[market]).abs().mean()) if len(ev) else float("nan")
        cal = calibrate(feats, market, season - 1, season)
        rows.append({"market": market, "n": cal["n"], "mae": round(mae, 2),
                     "cover80": cal["cover80"], "cover50": cal["cover50"],
                     "mean_pit": cal["mean_pit"]})
    return pd.DataFrame(rows)


def _print_report(r: dict) -> None:
    o, n = r["ours"], r["naive"]
    print(f"\nDraft-and-win backtest — {r['season']} — {r['sims']} sims, {r['teams']}-team league")
    print(f"  (fair title rate if drafting were a coin flip: {r['fair_title_rate']})\n")
    print(f"  {'':14}{'OUR board':>12}{'naive board':>14}")
    print(f"  {'title rate':14}{o['title_rate']:>12}{n['title_rate']:>14}")
    print(f"  {'playoff rate':14}{o['playoff_rate']:>12}{n['playoff_rate']:>14}")
    print(f"  {'mean finish':14}{o['mean_finish']:>12}{n['mean_finish']:>14}")
    print(f"  {'reg-season pts':14}{o['mean_points']:>12}{n['mean_points']:>14}")
    print(f"\n  Our value model's lift: {r['title_lift']:+} title rate, "
          f"{r['playoff_lift']:+} playoff rate.")


def main() -> None:
    import argparse
    from .scoring import HALF_PPR, STANDARD
    p = argparse.ArgumentParser(
        prog="python -m ffdata.backtest_draft",
        description="Retrospective draft-and-win backtest (leak-free draft, real replay).")
    p.add_argument("--season", type=int, required=True, help="season to replay")
    p.add_argument("--sims", type=int, default=200, help="random draft slots to simulate")
    p.add_argument("--teams", type=int, default=12)
    p.add_argument("--rounds", type=int, default=14, help="roster size (draft rounds)")
    p.add_argument("--scoring", choices=["ppr", "half", "standard"], default="ppr")
    p.add_argument("--props", action="store_true", help="also print prop-accuracy calibration")
    args = p.parse_args()
    rules = {"ppr": PPR, "half": HALF_PPR, "standard": STANDARD}[args.scoring]

    print(f"Backtesting {args.season} ({args.scoring.upper()})... building the preseason "
          "board and replaying the real season.", flush=True)
    _print_report(run_backtest(args.season, rules=rules, sims=args.sims,
                               teams=args.teams, rounds=args.rounds))
    if args.props:
        print("\nProp-guessing accuracy (out-of-sample calibration):")
        print(prop_accuracy(args.season).to_string(index=False))


if __name__ == "__main__":
    main()
