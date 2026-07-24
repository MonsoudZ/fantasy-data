"""Play a real past season blind: draft a team, then MANAGE it week to week.

`backtest_draft.py` answers "was the draft good?" -- it grades every roster with a
perfect-hindsight lineup (`best_week_total` picks starters using the points they
actually scored). That isolates draft value, but it is not a season anybody could
have played: nobody knows on Saturday who will score on Sunday.

This module answers the harder question: **would the app have won the league?**
Every decision is made with only what was knowable at the time.

    week w:  project    <- MatchupSimulator.project(season, w) trains on _k < w
             start      <- best lineup BY PROJECTION, for ALL 12 teams
             score      <- the points those starters actually scored
             waivers    <- every team, worst-first, adds a free agent if it's a
                           real upgrade (by FORM, not one noisy week)

Three separate walls against hindsight, each enforced in code rather than by
convention:

  * the draft sees `draft_board(season)`: prior-season features + preseason
    context (age, schedule, coaching), never a snap of the season itself;
  * `project(season, w)` fits on `_k < season*100 + w` -- literally every row
    before that kickoff and nothing after;
  * `project_kdst(season, w)` is a trailing mean over prior weeks only;
  * actual points are read ONLY to score a lineup that was already locked, and
    to run waivers for the FOLLOWING week.

EVERY team is managed the same way -- so our edge has to come from the draft, not
from being the only team that touches its roster (which the first version quietly
gave us). Two things made all-team management realistic rather than chaotic:
waivers value a player by his season-to-date FORM, so a stud on bye isn't dropped
for a streamer; and a move needs a real projected upgrade (`WAIVER_MIN_GAIN`), so
teams don't churn every week on ~6-RMSE noise. Without those, over-managing on
single-week projections circulated studs around the league on their byes and made
the title a coin flip -- a genuine finding, now guarded against.

    from ffdata.season_sim import run_season, format_league_report
    print(format_league_report(run_season(2024, detail=True)))   # full league
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_draft import _naive_board, round_robin
from .db import connect
from .kdst import build_dst, build_kicker, project_kdst, score_dst, score_kicker
from .optimize import _ELIGIBLE, _norm
from .scoring import HALF_PPR, PPR, STANDARD, ScoringRules, score

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
NFL_GAMES = 17                               # to prorate a season projection -> a week
# Weeks a rostered player can be absent from projections before he's treated as
# hurt/IR rather than on a bye. One missing week is a bye -- you ride it out; two
# or more means injured, and a real manager drops him for a live body.
INJURY_ABSENCE = 2
# Smallest projected-points upgrade to a starting slot worth a waiver move. Real
# managers have inertia; without this every team churns weekly on ~6-RMSE noise,
# which (via bye weeks) circulates studs around the league and turns the whole
# thing into a lottery. Measured effect of setting it: see the module finding.
WAIVER_MIN_GAIN = 3.0
# How many free agents per position each team actually considers on waivers. A
# team never rosters the 100th-best available WR, and run_waivers only adds a
# starting upgrade anyway, so the tail is dead weight -- capping it is realistic
# and keeps the all-team replay fast enough to sweep every draft slot.
WAIVER_POOL_PER_POS = 6
# Risk haircut on rookie DRAFT value. Rookie projections are ~calibrated on the
# mean but high-variance (most bust), so a veteran at the same number is safer in
# a H2H league. Below 1 makes the board prefer proven veterans; measured by the
# sweep. See CLAUDE.md's rookie finding.
ROOKIE_DRAFT_DISCOUNT = 0.80
# Lineup haircut on an UNPROVEN rookie (seeded from his preseason prior, no games
# yet). Below 1 means "start the veteran, hold the rookie" -- he only starts when
# he's clearly better or there's no veteran for the slot. Once he has real weekly
# projections he's judged on those, no discount.
ROOKIE_START_DISCOUNT = 0.60


def _seed_rookie_prior(wk_proj: dict, prior_wk: dict, ever_projected: set) -> dict:
    """This week's projections, with a preseason prior filled in for players the
    trailing model has NEVER seen (rookies / debuts with no games yet).

    The weekly projector is trailing, so a rookie is simply absent from it and
    would project 0 -- benched, or started blind. That penalises a rookie-heavy
    draft the season model rated highly. Seed those players from their preseason
    projection so the lineup can start them, as a real manager would on hype.

    Critically NOT applied to a player already in `ever_projected`: if he was
    projected in an earlier week and is absent now, he's on a bye or hurt, and
    must stay ~0 so he sits -- otherwise we'd start bye-week players and tank
    everyone's scores (measured: it dropped the league average from 74 to 67).
    """
    proj = dict(wk_proj)
    for k, v in prior_wk.items():
        if k not in proj and k not in ever_projected:
            proj[k] = v
    return proj


def _is_injured(key: str, ever_projected: set, absent_streak: dict) -> bool:
    """Is a rostered player hurt/IR (drop him) versus on a one-week bye (keep him)?

    A player the model has projected before, now absent from projections for
    INJURY_ABSENCE+ straight weeks, is injured -- a real manager drops him for a
    live body. A single missing week is a bye (ride it out). A player never yet
    projected is a pre-debut rookie, not hurt, so he's exempt.
    """
    return (key in ever_projected
            and absent_streak.get(key, 0) >= INJURY_ABSENCE)


def _beats(scores, bench, a, b, week) -> bool:
    """Does team `a` beat team `b` in `week`? Starters decide the game; a starter
    tie is broken by BENCH points (the user's league rule). `bench=None` skips the
    tiebreak (bench points unknown)."""
    if scores[a, week] != scores[b, week]:
        return bool(scores[a, week] > scores[b, week])
    if bench is not None and bench[a, week] != bench[b, week]:
        return bool(bench[a, week] > bench[b, week])
    return False        # genuinely tied on both -- caller decides (seed / half-win)


def standings_with_bench(scores, bench, sched) -> list[dict]:
    """Head-to-head W/L, but a tied starter score is broken by BENCH points.

    Only when the benches ALSO tie is a matchup a true draw (half a win each).
    Table is ordered by wins, then points-for (starters only -- bench never counts
    toward the season total, just toward breaking a single matchup).
    """
    n = scores.shape[0]
    wins = np.zeros(n)
    pf = scores[:, :len(sched)].sum(axis=1)
    for w, pairs in enumerate(sched):
        for a, b in pairs:
            if _beats(scores, bench, a, b, w):
                wins[a] += 1
            elif _beats(scores, bench, b, a, w):
                wins[b] += 1
            else:                       # tied on starters AND bench
                wins[a] += 0.5
                wins[b] += 0.5
    table = [{"team": t, "wins": float(wins[t]), "pf": float(pf[t])} for t in range(n)]
    table.sort(key=lambda r: (r["wins"], r["pf"]), reverse=True)
    return table


def playoff_bracket(seeds: list[int], scores, weeks: list[int],
                    bench=None) -> tuple[int, list]:
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

    A tied starter score is broken by BENCH points, then (if still level) by the
    better seed. Returns (champion, [(week, higher, lower, winner), ...]).
    """
    log = []

    def game(week, a, b):
        hi, lo = (a, b) if seeds.index(a) < seeds.index(b) else (b, a)
        # Starters, then bench, then the better seed advances.
        win = hi if not _beats(scores, bench, lo, hi, week) else lo
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
                slots=STARTERS, limits=LIMITS,
                min_gain: float = 0.0) -> tuple[list[dict], dict | None]:
    """Swap our worst benchable player for the best free agent, if it's an upgrade.

    "Upgrade" is measured the way the free-agent tab measures it: how much the
    STARTING lineup's projection improves. A better bench player is worth nothing
    if he never starts, so a swap only happens when the starting total moves.

    `proj` should be a player's FORM (season-to-date average), not a single week:
    a real manager doesn't drop a stud who happens to be on bye. `min_gain` is the
    smallest improvement worth a roster move -- above ~0 it stops teams churning
    every week chasing projection noise (~6 RMSE on any single week).
    """
    base = sum(proj.get(_norm(p["player"]), 0.0)
               for p in start_by_projection(roster, proj, slots))
    counts: dict[str, int] = {}
    for p in roster:
        counts[p["position"]] = counts.get(p["position"], 0) + 1

    best, best_gain = None, max(min_gain, 1e-9)   # never churn for nothing
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
                 con=None, rookie_discount: float = ROOKIE_DRAFT_DISCOUNT
                 ) -> tuple[list[dict], list[dict]]:
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
    board = draft_board(season, league, rules=rules, con=con,
                        rookie_discount=rookie_discount)
    kdst = _preseason_kdst(con, season, rules)
    # Keep VOR and auction $ on each record -- they ARE the reason a VOR draft
    # takes a player (highest value over replacement still available), so the
    # report can explain every pick.
    ours = [{"player": r["player"], "position": r["position"], "proj": float(r["proj"]),
             "vor": float(r["vor"]), "auction": int(r["auction"]),
             "player_id": r["player_id"]}
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


# Starting lineup this league fields, for roster-aware drafting.
_STARTER_NEED = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}
_FLEX_POS = ("RB", "WR", "TE")
# How much a bench-depth pick is discounted vs one that fills a starting slot.
# Below 1 makes the draft fill its 2RB/2WR/1TE/FLEX/QB/K/DEF starters before
# hoarding a 4th RB -- which is how you avoid an elite-RB / scrap-WR roster.
BENCH_DISCOUNT = 0.5


def _need_factor(position: str, counts: dict) -> float:
    """1.0 if this player would fill an open STARTING slot, else BENCH_DISCOUNT.

    A player fills a dedicated slot while his position is short of its starter
    count (RB<2, WR<2, TE<1, ...); once those are full he can still fill the one
    FLEX (RB/WR/TE) if it's open; beyond that he's bench depth and gets discounted
    so the draft pivots to a position that still needs a starter.
    """
    have = counts.get(position, 0)
    if have < _STARTER_NEED.get(position, 0):
        return 1.0
    if position in _FLEX_POS:
        flex_used = sum(max(0, counts.get(p, 0) - _STARTER_NEED[p]) for p in _FLEX_POS)
        if flex_used < 1:                     # the single flex is still open
            return 1.0
    return BENCH_DISCOUNT


def _roster_aware_draft(boards, need_aware, rounds, limits):
    """Snake draft where `need_aware` teams weight VOR by open starting slots.

    Non-need-aware teams take the best available on their (pre-sorted) board, as
    before. A need-aware team re-ranks each pick by VOR x `_need_factor`, so a 4th
    RB (bench) yields to an elite WR filling an empty WR slot.
    """
    from .backtest_draft import snake_order
    n = len(boards)
    taken: set[str] = set()
    counts = [{} for _ in range(n)]
    rosters: list[list[dict]] = [[] for _ in range(n)]
    for team in snake_order(n, rounds):
        pick = None
        if team in need_aware:
            best_val = -1e18
            for p in boards[team]:
                k = _norm(p["player"])
                pos = p["position"]
                if k in taken or counts[team].get(pos, 0) >= limits.get(pos, 99):
                    continue
                # K/DEF have no VOR and belong last; force them behind skill depth.
                base = -1e6 if pos in ("K", "DEF") else p.get("vor", 0.0)
                val = base * _need_factor(pos, counts[team])
                if val > best_val:
                    best_val, pick = val, p
        else:
            for p in boards[team]:
                k = _norm(p["player"])
                if k not in taken and counts[team].get(p["position"], 0) < limits.get(p["position"], 99):
                    pick = p
                    break
        if pick is None:
            continue
        taken.add(_norm(pick["player"]))
        counts[team][pick["position"]] = counts[team].get(pick["position"], 0) + 1
        rosters[team].append(pick)
    return rosters


def _jitter(name: str, team: int, scale: float) -> float:
    """Deterministic per-team noise on a player's draft value, in [-scale, scale].

    A stand-in for differing manager opinions -- reproducible (a hash, not RNG,
    which the sandbox forbids anyway) so the whole simulation stays deterministic.
    """
    import hashlib
    h = int(hashlib.md5(f"{name}|{team}".encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF - 0.5) * 2 * scale


def _draft_boards_for(ours, naive, n_teams, our_slot, opponent, noise):
    """The board each team drafts from.

    - "naive": opponents rank by last year's RAW points. They hoard QBs (a QB
      outscores any RB outright) and leave every elite RB/WR on the board, so a
      VOR drafter feasts. A weak, unrealistic field -- good for showing the board
      captures positional scarcity, bad for claiming we'd win a real league.
    - "sharp": opponents draft off the SAME VOR board we do, each reranked by its
      own `_jitter` (differing opinions). Now everyone drafts well and our only
      edge is being the *un-noised* board -- the honest, hard test.
    """
    if opponent == "sharp":
        boards = []
        for t in range(n_teams):
            if t == our_slot or noise <= 0:
                boards.append(ours)
            else:
                boards.append(sorted(ours, key=lambda p: -(p["proj"]
                                     + _jitter(p["player"], t, noise))))
        return boards
    return [ours if t == our_slot else naive for t in range(n_teams)]


def _overall_pick(rnd0: int, our_slot: int, n_teams: int) -> int:
    """1-based overall pick for our team in a snake draft, given the 0-based round."""
    if rnd0 % 2 == 0:                       # odd rounds (1,3,..) go left-to-right
        return rnd0 * n_teams + our_slot + 1
    return rnd0 * n_teams + (n_teams - our_slot)


def _draft_why(drafted, season, rules, con, our_slot, n_teams):
    """Explain every one of our picks: round, overall pick, value, and the reason.

    A VOR snake draft takes the highest value-over-replacement player still
    available that fits an open slot -- so VOR *is* the reason, and the situational
    context (rookie draft capital, a vacated role, a team change) is the colour.
    Returns one dict per pick, in draft order.

    NOTE: this is report colour ONLY -- it never touches the picks or the scores,
    which come from the leak-free VOR board. In a historical replay the injury/
    availability half of `player_context` reads the target season's own reports,
    so a "why" line can carry a little hindsight the drafter wouldn't have had.
    That's cosmetic; the measured finish and title rate are unaffected.
    """
    from .draft import player_context, rookie_context
    try:
        vctx = player_context(season, rules, con).set_index("player_id")
    except Exception:  # noqa: BLE001 - context is a bonus, never fatal
        vctx = None
    try:
        rk = {_norm(r["player"]): r for _, r in rookie_context(season, con=con).iterrows()}
    except Exception:  # noqa: BLE001
        rk = {}

    seen: dict[str, int] = {}
    out = []
    for i, p in enumerate(drafted):
        pos = p["position"]
        seen[pos] = seen.get(pos, 0) + 1
        why = []
        rrow = rk.get(_norm(p["player"]))
        if rrow is not None:
            why.append(f"rookie — draft pick #{int(rrow['pick'])}")
            if rrow.get("vacated_fp", 0) and rrow["vacated_fp"] > 40:
                why.append(f"{int(rrow['vacated_fp'])} vacated at {rrow['team']}")
        elif vctx is not None and p.get("player_id") in vctx.index:
            c = vctx.loc[p["player_id"]]
            if bool(c.get("moved")) and pd.notna(c.get("prior_team")):
                why.append(f"{c['prior_team']}→{c['team']}")
            blocked = c.get("blocked_by")
            if pd.notna(blocked) and blocked:
                why.append(f"behind {blocked}")
            elif pos in ("RB", "WR", "TE"):
                why.append("leads his room")
            vac = c.get("vacated_fp")
            if pd.notna(vac) and vac > 60:
                why.append(f"{int(vac)} vacated")
        out.append({
            "round": i + 1,
            "pick": _overall_pick(i, our_slot, n_teams),
            "player": p["player"], "position": pos,
            "pos_rank": seen[pos],                 # our Nth at this position
            "proj": round(p.get("proj", 0.0)),
            "vor": round(p.get("vor", 0.0)),
            "auction": p.get("auction"),
            "why": " · ".join(why),
        })
    return out


def _lineup_record(roster, proj, actual):
    """A week's lineup for one team: who STARTED (with slot) and who sat (bench)."""
    starters = start_by_projection(roster, proj, STARTERS)
    started = {_norm(p["player"]) for p in starters}
    bench = [{"player": p["player"], "position": p["position"],
              "proj": round(proj.get(_norm(p["player"]), 0.0), 1),
              "actual": round(actual.get(_norm(p["player"]), 0.0), 1)}
             for p in roster if _norm(p["player"]) not in started]
    return {
        "starters": [{"slot": p["slot"], "player": p["player"], "position": p["position"],
                      "proj": round(proj.get(_norm(p["player"]), 0.0), 1),
                      "actual": round(actual.get(_norm(p["player"]), 0.0), 1)}
                     for p in starters],
        "bench": sorted(bench, key=lambda b: -b["proj"]),
    }


def run_season(season: int, rules: ScoringRules = STANDARD, n_teams: int = 12,
               our_slot: int = 0, projector: str = "gbm", waivers: bool = True,
               con=None, log=print, ctx: dict | None = None, detail: bool = False,
               league_waivers: bool = True, opponent: str = "naive",
               noise: float = 24.0) -> dict:
    """Draft blind, manage every week on projections only, report the whole league.

    EVERY team is managed the same way -- best lineup by projection each week, and
    a waiver claim if a free agent would raise its starting total -- so our edge
    has to come from the draft, not from being the only team that touches its
    roster (which is what the earlier version quietly gave us).

    Waivers run in priority order: worst team so far claims first, standard league
    rules. That both resolves who lands a contested free agent and stops the same
    player being added by two teams in one week.

    `detail=True` keeps every team's weekly starters/bench and transaction log --
    heavy, for a single inspectable season. `run_all_slots` leaves it off.
    """
    ctx = ctx or prepare(season, rules, projector, n_teams, con, log)
    ours, naive = ctx["ours"], ctx["naive"]
    project, by_week, n_teams = ctx["project"], ctx["by_week"], ctx["n_teams"]

    log(f"Snake draft: {n_teams} teams x {ROSTER_SIZE} rounds, we pick at slot {our_slot + 1}")
    boards = _draft_boards_for(ours, naive, n_teams, our_slot, opponent, noise)
    # We draft to roster need (fill 2RB/2WR/1TE/FLEX/QB/K/DEF starters before
    # bench depth), so we don't hoard RBs and end up with scrap WRs. A sharp field
    # drafts the same way (competent managers balance); the naive field keeps its
    # defining flaw -- raw-points greed that hoards QBs.
    need_aware = set(range(n_teams)) if opponent == "sharp" else {our_slot}
    rosters = _roster_aware_draft(boards, need_aware, ROSTER_SIZE, LIMITS)
    drafted_rosters = [[dict(p) for p in r] for r in rosters]

    # Preseason per-week fallback, from the (leak-free) draft board. The weekly
    # projector is TRAILING -- a rookie with no games yet is simply absent from
    # its board, so proj=0 and the lineup benches him or starts him blind. That
    # penalises a rookie-heavy draft the model itself rated highly (2025: we drew
    # Jeanty, Hunter, McMillan, Egbuka, Golden...). A real manager starts a hyped
    # rookie on preseason expectation, so seed a missing player's week from his
    # season projection / a full season. Used ONLY when the weekly model has
    # nothing; once he has trailing games, the weekly number takes over.
    prior_wk = {_norm(p["player"]): p["proj"] / NFL_GAMES
                for p in ours + naive if p.get("proj")}

    weeks = list(REG_WEEKS) + list(PLAYOFF_WEEKS)
    scores = np.zeros((n_teams, len(weeks)))       # STARTER points -- decide the game
    bench = np.zeros((n_teams, len(weeks)))        # BENCH points -- tiebreak only
    taken = {_norm(p["player"]) for r in rosters for p in r}
    txns: list[list[dict]] = [[] for _ in range(n_teams)]
    lineups: list[list[dict]] = [[] for _ in range(n_teams)]
    # A running mean of each player's weekly projections -- his "form" to date.
    # Waivers decide on THIS, not the single upcoming week, so a stud on bye
    # (who projects ~0 next week) isn't dropped for a streamer. Leak-free: it
    # only ever averages projections already computed, never results.
    psum: dict[str, float] = {}
    pcnt: dict[str, int] = {}
    # Players the trailing model has EVER projected. A player absent this week who
    # is in here is on a bye / injured -> he must stay ~0 and sit, NOT get his
    # preseason prior (that would start bye-week players and tank scores). The
    # prior fallback is only for players who have never been seen: rookies/debuts.
    ever_projected: set[str] = set()
    # Consecutive weeks each rostered player has had no game. 1 = bye (keep him),
    # >= INJURY_ABSENCE = hurt/IR (waivers drop him for a live body).
    absent_streak: dict[str, int] = {}

    def value_of(key: str, this_week: float) -> float:
        """Season-to-date average projection, seeded by this week's number."""
        if key in pcnt:
            return (psum[key] + this_week) / (pcnt[key] + 1)
        return this_week

    for wi, w in enumerate(weeks):
        if w not in by_week:
            log(f"  week {w}: no results in the lake, stopping here")
            weeks = weeks[:wi]
            scores = scores[:, :wi]
            bench = bench[:, :wi]
            break
        wk_proj, _ = project(w)
        proj = _seed_rookie_prior(wk_proj, prior_wk, ever_projected)
        ever_projected.update(wk_proj)
        # A rostered player with no game this week is on a bye or hurt. Count the
        # streak so waivers can tell one-week byes (ride out) from injuries (drop).
        active = set(wk_proj)
        for r in rosters:
            for p in r:
                k = _norm(p["player"])
                absent_streak[k] = 0 if k in active else absent_streak.get(k, 0) + 1
        actual_w = by_week[w]
        for k, v in proj.items():           # fold this week into each player's form
            psum[k] = psum.get(k, 0.0) + v
            pcnt[k] = pcnt.get(k, 0) + 1
        # LINEUP projection: an unproven rookie (seeded from his preseason prior,
        # no games yet) is discounted so a veteran with a real projection starts
        # ahead of him -- "start the veteran, hold the rookie". He still starts
        # when there's no veteran for the slot, or when even discounted he's best.
        seeded = set(proj) - set(wk_proj)
        lineup_proj = {k: (v * ROOKIE_START_DISCOUNT if k in seeded else v)
                       for k, v in proj.items()}

        for t in range(n_teams):
            starters = start_by_projection(rosters[t], lineup_proj, STARTERS)
            scores[t, wi] = week_score(starters, actual_w)
            # Bench = everyone rostered but not started. Only ever used to break a
            # tie in a head-to-head matchup (the user's league rule); it never
            # counts toward a team's score otherwise.
            started = {_norm(p["player"]) for p in starters}
            bench[t, wi] = round(sum(actual_w.get(_norm(p["player"]), 0.0)
                                     for p in rosters[t]
                                     if _norm(p["player"]) not in started), 2)
            if detail or t == our_slot:
                rec = _lineup_record(rosters[t], lineup_proj, actual_w)
                rec.update(week=w, points=scores[t, wi], bench_points=bench[t, wi])
                lineups[t].append(rec)

        # Waivers run AFTER the week is scored, on NEXT week's projection -- the
        # order a real league runs them. Worst-team-first priority.
        if waivers and wi + 1 < len(weeks):
            nxt, nxt_pool = project(weeks[wi + 1])
            # Value every roster/pool player by his form, not the single upcoming
            # week -- and only over players actually available or rostered. A
            # player gone 2+ weeks is hurt/IR: zero his value so a real manager
            # drops him for a live body (a one-week bye keeps his form). Pre-debut
            # rookies (never projected) are exempt -- they haven't been hurt.
            wval = {}
            for k in set(nxt) | taken:
                if _is_injured(k, ever_projected, absent_streak):
                    wval[k] = 0.0
                else:
                    wval[k] = value_of(k, nxt.get(k, 0.0))
            # Only the best free agents matter: run_waivers requires a starting-
            # lineup upgrade, so a replacement-level FA is never added. Keeping the
            # top few per position by form is realistic (no one rosters the 100th
            # WR) and ~5x faster than scanning all ~230 free agents each team-week.
            ranked_fa = sorted(((p, pos) for p, pos in nxt_pool),
                               key=lambda x: -wval.get(_norm(x[0]), 0.0))
            per_pos: dict[str, int] = {}
            shortlist = []
            for p, pos in ranked_fa:
                if per_pos.get(pos, 0) < WAIVER_POOL_PER_POS:
                    shortlist.append((p, pos))
                    per_pos[pos] = per_pos.get(pos, 0) + 1
            # league_waivers=False reproduces the earlier behaviour (only our
            # team manages its roster) for an apples-to-apples comparison.
            order = _waiver_order(scores[:, :wi + 1]) if league_waivers else [our_slot]
            for t in order:
                pool = [{"player": p, "position": pos} for p, pos in shortlist
                        if _norm(p) not in taken]
                rosters[t], mv = run_waivers(rosters[t], pool, wval,
                                             min_gain=WAIVER_MIN_GAIN)
                if mv:
                    mv["week"] = weeks[wi + 1]
                    taken.add(_norm(mv["add"]))
                    taken.discard(_norm(mv["drop"]))
                    if detail or t == our_slot:
                        txns[t].append(mv)
        log(f"  week {w}: we scored {scores[our_slot, wi]:.1f}")

    sched = round_robin(n_teams, len(REG_WEEKS))
    reg_w = len(REG_WEEKS)
    table = standings_with_bench(scores[:, :reg_w], bench[:, :reg_w], sched)
    seeds = [row["team"] for row in table][:6]
    pw = [weeks.index(w) for w in PLAYOFF_WEEKS if w in weeks]
    champ, bracket = (playoff_bracket(seeds, scores, pw, bench) if len(pw) == 3
                      else (seeds[0], []))
    place = next((i + 1 for i, row in enumerate(table) if row["team"] == our_slot), None)

    return {"season": season, "our_slot": our_slot, "roster": rosters[our_slot],
            "drafted": drafted_rosters[our_slot], "drafted_all": drafted_rosters,
            "draft_why": (_draft_why(drafted_rosters[our_slot], season, rules, con,
                                     our_slot, n_teams) if detail else None),
            "n_teams": n_teams,
            "final_rosters": rosters, "weeks": lineups[our_slot],
            "league_lineups": lineups if detail else None,
            "moves": txns[our_slot], "league_txns": txns if detail else None,
            "standings": table, "seeds": seeds, "bracket": bracket,
            "regular_season_place": place, "champion": champ,
            "we_won": champ == our_slot, "scores": scores, "week_numbers": weeks}


def _waiver_order(scores_so_far) -> list[int]:
    """Waiver priority: fewest points so far claims first (standard worst-first)."""
    totals = scores_so_far.sum(axis=1)
    return list(np.argsort(totals, kind="stable"))


def run_all_slots(season: int, rules: ScoringRules = STANDARD, n_teams: int = 12,
                  projector: str = "gbm", waivers: bool = True, con=None,
                  log=print, opponent: str = "naive", noise: float = 24.0) -> dict:
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
                       con, log=lambda *a: None, ctx=ctx,
                       opponent=opponent, noise=noise)
        runs.append(r)
        log(f"  slot {slot + 1:2d}: finished {r['regular_season_place']:2d} of {n_teams}"
            f"  ({'CHAMPION' if r['we_won'] else 'no title'})")
    places = [r["regular_season_place"] for r in runs]
    return {"season": season, "runs": runs, "places": places,
            "titles": sum(r["we_won"] for r in runs),
            "mean_place": round(sum(places) / len(places), 2),
            "playoff_rate": round(sum(p <= 6 for p in places) / len(places), 3)}


def _team_label(t: int, our_slot: int) -> str:
    return f"Team {t + 1}" + (" (US)" if t == our_slot else "")


def format_league_report(r: dict, sample_weeks=(1, 9, 17)) -> str:
    """Human-readable full-league report for a `detail=True` run.

    Shows the final standings, our draft and how it changed, every team's final
    roster split into starters-calibre and bench, and the transaction log -- the
    "who's on each team, who started, who got dropped" the summary can't convey.
    """
    L, our = [], r["our_slot"]
    place = r["regular_season_place"]
    champ = r["champion"]
    L.append(f"=== {r['season']} — drafted from slot {our + 1}, "
             f"finished {place} of {len(r['standings'])} "
             f"({'CHAMPION' if r['we_won'] else 'team ' + str(champ + 1) + ' won'}) ===")

    L.append("\nFinal standings (regular season):")
    games = len(REG_WEEKS)
    for i, row in enumerate(r["standings"]):
        tag = "  <-- US" if row["team"] == our else ""
        seed = " [playoffs]" if row["team"] in r["seeds"] else ""
        wins = int(row["wins"])
        L.append(f"  {i + 1:2d}. {_team_label(row['team'], our):14s} "
                 f"{wins:2d}-{games - wins:<2d}  {row['pf']:7.1f} PF{seed}{tag}")

    # Our team in detail -- each pick with the value that drove it and why.
    if r.get("draft_why"):
        L.append("\nOUR DRAFT — blind off the preseason board. Each pick is the "
                 "highest VOR still available that fits an open slot:")
        L.append(f"  {'Rd':>2} {'Pk':>3}  {'Pos':4} {'Player':22} {'Proj':>4} "
                 f"{'VOR':>5} {'$':>3}   Why")
        for d in r["draft_why"]:
            posrank = f"{d['position']}{d['pos_rank']}"
            why = d["why"] or f"best {d['position']} left"
            aud = "" if d["auction"] is None else f"{d['auction']:>3}"
            L.append(f"  {d['round']:>2} {d['pick']:>3}  {posrank:4} {d['player']:22} "
                     f"{d['proj']:>4} {d['vor']:>+5} {aud:>3}   {why}")
    else:
        L.append("\nOUR DRAFT (blind, off the preseason board):")
        for i, p in enumerate(r["drafted"]):
            L.append(f"  R{i + 1:2d}  {p['position']:4s} {p['player']}")
    if r["moves"]:
        L.append("\nOUR TRANSACTIONS (waiver add / drop, on projection):")
        for m in r["moves"]:
            L.append(f"  wk{m['week']:2d}  + {m['add']:24s} - {m['drop']:24s} "
                     f"(+{m['gain']:.1f} proj to starters)")

    # A few weekly lineups so "who started" is concrete.
    if r["weeks"]:
        L.append("\nOUR LINEUPS (sampled weeks):")
        for wk in r["weeks"]:
            if wk["week"] not in sample_weeks:
                continue
            bp = wk.get("bench_points", 0.0)
            L.append(f"  Week {wk['week']} — {wk['points']:.1f} pts "
                     f"(bench {bp:.1f}, counts only to break a tie)")
            for s in wk["starters"]:
                L.append(f"     {s['slot']:5s} {s['player']:22s} "
                         f"proj {s['proj']:5.1f}  actual {s['actual']:5.1f}")
            if wk["bench"]:
                names = ", ".join(f"{b['player']} ({b['actual']:.0f})" for b in wk["bench"])
                L.append(f"     bench: {names}")

    # Who every team took, round by round (the draft as it happened).
    if r.get("drafted_all"):
        L.append("\nDRAFT BOARD — every team's pick each round (snake order):")
        for t, roster in enumerate(r["drafted_all"]):
            picks = " | ".join(f"{p['position']} {p['player']}" for p in roster)
            L.append(f"  {_team_label(t, our):14s}: {picks}")

    # Every team's roster AS IT ENDED, after a season of waivers.
    if r.get("final_rosters"):
        L.append("\nEVERY TEAM'S FINAL ROSTER (after the season's waivers):")
        for t, roster in enumerate(r["final_rosters"]):
            names = ", ".join(f"{p['player']} ({p['position']})" for p in roster)
            L.append(f"  {_team_label(t, our):14s}: {names}")

    return "\n".join(L)


def main() -> None:
    import argparse

    from .ingest import current_nfl_season

    p = argparse.ArgumentParser(
        prog="python -m ffdata.season_sim",
        description="Play a real past season blind: draft on prior-year data, then "
                    "manage every team on projections only. Reports the whole league.")
    p.add_argument("--season", type=int, default=current_nfl_season(),
                   help="a PLAYED season to replay (needs its weekly results)")
    p.add_argument("--slot", type=int, default=1, help="our draft slot, 1-based")
    p.add_argument("--scoring", choices=["ppr", "half", "standard"], default="standard")
    p.add_argument("--all-slots", action="store_true",
                   help="replay from every draft slot and print the finish distribution")
    args = p.parse_args()

    from .ingest import season_not_started
    if season_not_started(args.season):
        raise SystemExit(f"{args.season} hasn't been played yet -- pick a finished season.")

    rules = {"ppr": PPR, "half": HALF_PPR, "standard": STANDARD}[args.scoring]

    if args.all_slots:
        r = run_all_slots(args.season, rules=rules)
        p_ = r["places"]
        print(f"\n{args.season}: mean finish {r['mean_place']}/12 · "
              f"playoffs {r['playoff_rate']:.0%} · titles {r['titles']}/12 · places {p_}")
        return

    r = run_season(args.season, rules=rules, our_slot=args.slot - 1, detail=True)
    print("\n" + format_league_report(r))


if __name__ == "__main__":
    main()
