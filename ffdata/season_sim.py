"""Play a real past season blind: draft a team, then MANAGE it week to week.

`backtest_draft.py` answers "was the draft good?" -- it grades every roster with a
perfect-hindsight lineup (`best_week_total` picks starters using the points they
actually scored). That isolates draft value, but it is not a season anybody could
have played: nobody knows on Saturday who will score on Sunday.

This module answers the harder question: **would the app have won the league?**
Every decision is made with only what was knowable at the time.

    week w:  project    <- MatchupSimulator.project(season, w) trains on _k < w
             start      <- best lineup BY PROJECTION
             score      <- the points those starters actually scored
             waivers    <- swap in a free agent only if projected better

Three separate walls against hindsight, each enforced in code rather than by
convention:

  * the draft sees `draft_board(season)`: prior-season features + preseason
    context (age, schedule, coaching), never a snap of the season itself;
  * `project(season, w)` fits on `_k < season*100 + w` -- literally every row
    before that kickoff and nothing after;
  * `project_kdst(season, w)` is a trailing mean over prior weeks only;
  * actual points are read ONLY to score a lineup that was already locked, and
    to run waivers for the FOLLOWING week.

Because the whole league is managed the same way, the comparison is fair: our
team's edge has to come from better projections and better roster decisions, not
from a rule the opponents don't get.

    from ffdata.season_sim import run_season
    print(run_season(2024))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_draft import _naive_board, round_robin, run_snake_draft, standings
from .db import connect
from .kdst import build_dst, build_kicker, project_kdst, score_dst, score_kicker
from .optimize import _ELIGIBLE, _norm
from .scoring import STANDARD, ScoringRules, score

# 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 1 DEF, 1 K.
STARTERS = ("QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "DEF", "K")
BENCH = 5
ROSTER_SIZE = len(STARTERS) + BENCH          # 14
# Caps sum to exactly ROSTER_SIZE, which is what forces a kicker and a defense.
# run_snake_draft takes the best uncapped player available, so if skill positions
# could absorb all 14 picks it would simply never reach K/DST at the bottom of
# the board -- the same reason real drafts leave them to the last two rounds.
LIMITS = {"QB": 2, "RB": 4, "WR": 4, "TE": 2, "DEF": 1, "K": 1}
REG_WEEKS = tuple(range(1, 15))              # weeks 1-14
PLAYOFF_WEEKS = (15, 16, 17)                 # 6-team bracket, top 2 get a bye


def playoff_bracket(seeds: list[int], scores, weeks: list[int]) -> tuple[int, list]:
    """Standard 6-team bracket: the top two seeds get a first-round bye.

    `backtest_draft.playoffs` pairs the whole field every round, so a six-team
    bracket makes the 1 seed win three straight single-elimination games -- and a
    title becomes very nearly a coin flip regardless of how the regular season
    went. Measured on 2024: with no byes we won 1 of 12 runs (8.3%, exactly the
    1-in-12 base rate) despite finishing first six times. Byes are what make the
    regular season worth playing.

        wk 15  quarterfinals   3v6, 4v5      (seeds 1-2 idle)
        wk 16  semifinals      1 v lowest, 2 v other
        wk 17  final

    Returns (champion, [(week, higher, lower, winner), ...]).
    """
    log = []

    def game(week, a, b):
        hi, lo = (a, b) if seeds.index(a) < seeds.index(b) else (b, a)
        win = hi if scores[hi, week] >= scores[lo, week] else lo   # tie -> better seed
        log.append((week, hi, lo, win))
        return win

    qf = [game(weeks[0], seeds[2], seeds[5]), game(weeks[0], seeds[3], seeds[4])]
    alive = sorted(qf, key=seeds.index)
    sf = [game(weeks[1], seeds[0], alive[1]),      # 1 seed draws the WORSE survivor
          game(weeks[1], seeds[1], alive[0])]
    champ = game(weeks[2], *sorted(sf, key=seeds.index))
    return champ, log


def start_by_projection(roster: list[dict], proj: dict, slots=STARTERS) -> list[dict]:
    """Choose starters using PROJECTED points -- the lineup you'd actually set.

    Same greedy slot-fill the optimizer uses, but ranked by projection rather
    than by the result. A player with no projection (bye, injured, not in the
    model's universe) sorts to the bottom and only starts if nothing else fits.
    """
    ranked = sorted(roster, key=lambda p: -proj.get(_norm(p["player"]), -1.0))
    used, lineup = set(), []
    for slot in slots:
        for p in ranked:
            key = _norm(p["player"])
            if key not in used and p["position"] in _ELIGIBLE[slot]:
                used.add(key)
                lineup.append({**p, "slot": slot})
                break
    return lineup


def week_score(lineup: list[dict], actual: dict) -> float:
    """What a locked lineup actually scored."""
    return round(sum(actual.get(_norm(p["player"]), 0.0) for p in lineup), 2)


def _dst_name(name: str, position: str) -> str:
    """One spelling for a defense, everywhere.

    `project_kdst` calls it "PHI DST"; `build_dst` calls it "PHI". Left alone, a
    drafted defense never matches its own projection -- it scores zero every week
    and waivers churn the slot chasing a player who cannot score. Kickers already
    agree (both use the display name), so only DEF needs bridging.
    """
    if position != "DEF":
        return name
    return name if name.upper().endswith(" DST") else f"{name} DST"


def _preseason_kdst(con, season: int, rules: ScoringRules) -> list[dict]:
    """Draftable K/DST, valued by their PRIOR season's total.

    The season model only ranks QB/RB/WR/TE, but this league starts a kicker and
    a defense, so they have to come from somewhere. Last year's total is the only
    preseason signal available for them -- and, per the K/DST honesty note, about
    as predictive as anything else for these two positions.
    """
    out = []
    for build, scorer, key, pos in ((build_kicker, score_kicker, "player_display_name", "K"),
                                    (build_dst, score_dst, "team", "DEF")):
        try:
            df = build(con)
        except Exception:  # noqa: BLE001 - a lake without K/DST just drafts fewer
            continue
        if df.empty:
            continue
        df = scorer(df[df["season"] == season - 1], rules, col="fp")
        tot = df.groupby(key, as_index=False)["fp"].sum().sort_values("fp", ascending=False)
        out += [{"player": _dst_name(str(r[key]), pos), "position": pos,
                 "proj": float(r["fp"])} for _, r in tot.iterrows()]
    return out


def _actual_points(con, season: int, rules: ScoringRules) -> pd.DataFrame:
    """Every player's REAL points per week -- skill players, kickers, defenses.

    Read only to grade lineups that are already locked. Never reaches the draft
    or the projections.
    """
    frames = []
    wk = con.sql("select * from weekly where season = ? and season_type = 'REG'",
                 params=[season]).df()
    skill = score(wk[wk["position"].isin(["QB", "RB", "WR", "TE"])], rules, col="fp")
    frames.append(skill[["player_display_name", "week", "fp"]]
                  .rename(columns={"player_display_name": "player"}))

    kick = score_kicker(wk[wk["position"] == "K"], rules, col="fp")
    if not kick.empty:
        frames.append(kick[["player_display_name", "week", "fp"]]
                      .rename(columns={"player_display_name": "player"}))
    try:
        dst = build_dst(con)
        dst = score_dst(dst[dst["season"] == season], rules, col="fp")
        if not dst.empty:
            dst = dst.assign(player=dst["team"].map(lambda t: _dst_name(str(t), "DEF")))
            frames.append(dst[["player", "week", "fp"]])
    except Exception:  # noqa: BLE001
        pass
    return pd.concat(frames, ignore_index=True)


class WeekProjections:
    """Every week's projection, computed once and memoised.

    `sim.project()` refits the model on each call (that refit is exactly what
    keeps it leak-free -- it trains on `_k < season*100 + week` and nothing
    after), so calling it three times a week would triple the cost for identical
    output. Cache per week; the leak-free property is unchanged.
    """

    def __init__(self, sim, season: int, rules: ScoringRules, con):
        self.sim, self.season, self.rules, self.con = sim, season, rules, con
        self._cache: dict[int, tuple[dict, list]] = {}

    def __call__(self, week: int) -> tuple[dict, list]:
        """(player -> projected points, [(player, position)] pool) for `week`."""
        if week not in self._cache:
            proj, pool = {}, []
            board = self.sim.project(self.season, week)
            for _, r in board.iterrows():
                proj[_norm(r["player_display_name"])] = float(r["pred"])
                pool.append((r["player_display_name"], r["position"]))
            kd = project_kdst(self.season, week, rules=self.rules, con=self.con)
            for _, r in kd.iterrows():
                proj[_norm(r["player_display_name"])] = float(r["pred"])
                pool.append((r["player_display_name"], r["position"]))
            self._cache[week] = (proj, pool)
        return self._cache[week]


def run_waivers(roster: list[dict], pool: list[dict], proj: dict,
                slots=STARTERS, limits=LIMITS) -> tuple[list[dict], dict | None]:
    """Swap our worst benchable player for the best free agent, if it's an upgrade.

    "Upgrade" is measured the way the free-agent tab measures it: how much the
    STARTING lineup's projection improves. A better bench player is worth nothing
    if he never starts, so a swap only happens when the starting total moves.
    """
    base = sum(proj.get(_norm(p["player"]), 0.0)
               for p in start_by_projection(roster, proj, slots))
    counts: dict[str, int] = {}
    for p in roster:
        counts[p["position"]] = counts.get(p["position"], 0) + 1

    best, best_gain = None, 1e-9          # strictly positive: never churn for nothing
    on_roster = {_norm(p["player"]) for p in roster}
    for fa in pool:
        if _norm(fa["player"]) in on_roster:
            continue
        for i, drop in enumerate(roster):
            if drop["position"] != fa["position"]:
                # Position caps only bind when the swap changes the mix.
                if counts.get(fa["position"], 0) >= limits.get(fa["position"], 99):
                    continue
            trial = roster[:i] + roster[i + 1:] + [fa]
            gain = sum(proj.get(_norm(p["player"]), 0.0)
                       for p in start_by_projection(trial, proj, slots)) - base
            if gain > best_gain:
                best, best_gain = (trial, {"add": fa["player"], "drop": drop["player"],
                                           "gain": round(gain, 2)}), gain
    return (best[0], best[1]) if best else (roster, None)


def draft_boards(season: int, rules: ScoringRules = STANDARD, n_teams: int = 12,
                 con=None) -> tuple[list[dict], list[dict]]:
    """(our board, the naive opponents' board) -- preseason only, no projector.

    Split out of `prepare` because it costs seconds while fitting the weekly
    projector costs minutes, and the draft is testable on its own.

    Opponents rank by last season's raw points, the standard baseline here. Worth
    knowing what that does to a draft: raw points ignore positional scarcity, so
    naive teams reach for quarterbacks (a QB outscores an RB outright) while our
    VOR board spends early picks on the scarce positions.
    """
    con = con or connect()
    from .draft import draft_board

    league = {"teams": n_teams, "budget": 200, "roster_spots": ROSTER_SIZE,
              "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flex": 1}
    board = draft_board(season, league, rules=rules, con=con)
    kdst = _preseason_kdst(con, season, rules)
    ours = [{"player": r["player"], "position": r["position"], "proj": float(r["proj"])}
            for _, r in board.iterrows()] + kdst
    return ours, _naive_board(con, season, rules) + kdst


def prepare(season: int, rules: ScoringRules = STANDARD, projector: str = "gbm",
            n_teams: int = 12, con=None, log=print) -> dict:
    """Everything that doesn't depend on which draft slot we hold.

    The board, the fitted projector and each week's projections are identical for
    all 12 slots, and they are ~all of the cost -- so build them once and let
    every slot replay against the same league.
    """
    con = con or connect()
    from .matchup import MatchupSimulator

    log(f"Building the {season} draft board (prior seasons only)…")
    ours, naive = draft_boards(season, rules, n_teams, con)

    log(f"Fitting the weekly projector ({projector})…")
    sim = MatchupSimulator.fit(projector=projector, rules=rules)
    project = WeekProjections(sim, season, rules, con)
    actual = _actual_points(con, season, rules)
    by_week = {w: dict(zip(actual.loc[actual.week == w, "player"].map(_norm),
                           actual.loc[actual.week == w, "fp"]))
               for w in sorted(actual["week"].unique())}
    return {"con": con, "ours": ours, "naive": naive, "project": project,
            "by_week": by_week, "season": season, "rules": rules, "n_teams": n_teams}


def run_season(season: int, rules: ScoringRules = STANDARD, n_teams: int = 12,
               our_slot: int = 0, projector: str = "gbm", waivers: bool = True,
               con=None, log=print, ctx: dict | None = None) -> dict:
    """Draft blind, manage every week on projections only, report where we finish.

    Returns the full record: roster, weekly lineups and scores, standings, and
    the playoff result.
    """
    ctx = ctx or prepare(season, rules, projector, n_teams, con, log)
    ours, naive = ctx["ours"], ctx["naive"]
    project, by_week, n_teams = ctx["project"], ctx["by_week"], ctx["n_teams"]

    log(f"Snake draft: {n_teams} teams x {ROSTER_SIZE} rounds, we pick at slot {our_slot + 1}")
    boards = [ours if t == our_slot else naive for t in range(n_teams)]
    rosters = run_snake_draft(boards, ROSTER_SIZE, LIMITS)
    drafted_roster = [dict(p) for p in rosters[our_slot]]

    weeks = list(REG_WEEKS) + list(PLAYOFF_WEEKS)
    scores = np.zeros((n_teams, len(weeks)))
    moves, our_weeks = [], []
    drafted = {_norm(p["player"]) for r in rosters for p in r}

    for wi, w in enumerate(weeks):
        if w not in by_week:
            log(f"  week {w}: no results in the lake, stopping here")
            weeks = weeks[:wi]
            scores = scores[:, :wi]
            break
        proj, _ = project(w)
        actual_w = by_week[w]

        for t in range(n_teams):
            lineup = start_by_projection(rosters[t], proj, STARTERS)
            scores[t, wi] = week_score(lineup, actual_w)
            if t == our_slot:
                our_weeks.append({"week": w, "points": scores[t, wi],
                                  "lineup": [{"slot": p["slot"], "player": p["player"],
                                              "proj": round(proj.get(_norm(p["player"]), 0.0), 1),
                                              "actual": round(actual_w.get(_norm(p["player"]), 0.0), 1)}
                                             for p in lineup]})
        # Waivers run AFTER the week is scored, on next week's projection --
        # the same order a real league runs them.
        if waivers and wi + 1 < len(weeks):
            nxt, nxt_pool = project(weeks[wi + 1])
            pool = [{"player": p, "position": pos} for p, pos in nxt_pool
                    if _norm(p) not in drafted]
            rosters[our_slot], mv = run_waivers(rosters[our_slot], pool, nxt)
            if mv:
                mv["week"] = weeks[wi + 1]
                moves.append(mv)
                drafted.add(_norm(mv["add"]))
                drafted.discard(_norm(mv["drop"]))
        log(f"  week {w}: we scored {scores[our_slot, wi]:.1f}")

    sched = round_robin(n_teams, len(REG_WEEKS))
    reg = scores[:, :len(REG_WEEKS)]
    table = standings(reg, sched)
    seeds = [row["team"] for row in table][:6]
    pw = [weeks.index(w) for w in PLAYOFF_WEEKS if w in weeks]
    champ, bracket = (playoff_bracket(seeds, scores, pw) if len(pw) == 3
                      else (seeds[0], []))
    place = next((i + 1 for i, row in enumerate(table) if row["team"] == our_slot), None)

    return {"season": season, "our_slot": our_slot, "roster": rosters[our_slot],
            "drafted": drafted_roster,
            "weeks": our_weeks, "moves": moves, "standings": table,
            "regular_season_place": place, "champion": champ,
            "we_won": champ == our_slot, "scores": scores, "week_numbers": weeks}


def run_all_slots(season: int, rules: ScoringRules = STANDARD, n_teams: int = 12,
                  projector: str = "gbm", waivers: bool = True, con=None,
                  log=print) -> dict:
    """Replay the season from EVERY draft slot and report the distribution.

    One season from one slot is a single sample: draft position and the
    round-robin schedule swing a fantasy year enormously, and reporting whichever
    slot happened to win would be picking a winner after the fact. Running all
    twelve against the identical league turns one real season into a spread --
    a finish distribution and a title rate, not an anecdote.
    """
    ctx = prepare(season, rules, projector, n_teams, con, log)
    runs = []
    for slot in range(n_teams):
        r = run_season(season, rules, n_teams, slot, projector, waivers,
                       con, log=lambda *a: None, ctx=ctx)
        runs.append(r)
        log(f"  slot {slot + 1:2d}: finished {r['regular_season_place']:2d} of {n_teams}"
            f"  ({'CHAMPION' if r['we_won'] else 'no title'})")
    places = [r["regular_season_place"] for r in runs]
    return {"season": season, "runs": runs, "places": places,
            "titles": sum(r["we_won"] for r in runs),
            "mean_place": round(sum(places) / len(places), 2),
            "playoff_rate": round(sum(p <= 6 for p in places) / len(places), 3)}
