"""Blind season replay: lineups from projections, never from results.

The whole point of this module is that no decision may see the future. These
tests pin that property on synthetic frames (so they run without a lake) plus a
couple of real-data checks on the joins that silently produce zeros when broken.
"""

import pytest

from conftest import requires_data_lake
from ffdata.season_sim import (
    LIMITS, ROSTER_SIZE, STARTERS, _dst_name, playoff_bracket, run_waivers,
    start_by_projection, week_score,
)


def _p(name, pos):
    return {"player": name, "position": pos}


def _proj(*vals):
    """Projections keyed exactly as `_norm` produces them, in ROSTER order."""
    keys = ["ace quarter", "bench quarter", "cal runner", "dex runner", "eli runner",
            "fay wide", "gil wide", "hal wide", "ivy tight", "jay kicker", "kc dst"]
    return {k: v for k, v in zip(keys, vals) if v is not None}


# Distinct ALPHABETIC names on purpose: `_norm` strips digits, so "QB1"/"QB2"
# would collide into one key and every projection lookup would fool the test.
ROSTER = [_p("Ace Quarter", "QB"), _p("Bench Quarter", "QB"),
          _p("Cal Runner", "RB"), _p("Dex Runner", "RB"), _p("Eli Runner", "RB"),
          _p("Fay Wide", "WR"), _p("Gil Wide", "WR"), _p("Hal Wide", "WR"),
          _p("Ivy Tight", "TE"), _p("Jay Kicker", "K"), _p("KC DST", "DEF")]

def test_lineup_is_set_by_projection_not_by_what_happened():
    """The heart of it: a player who is projected badly but scores 40 must stay
    on the bench, because on Saturday you didn't know. If this ever inverts, the
    whole simulation becomes a hindsight oracle and its results are worthless."""
    proj = _proj(20, 5, 15, 12, 1, 14, 11, 2, 8, 8, 7)
    lineup = start_by_projection(ROSTER, proj, STARTERS)
    started = {p["player"] for p in lineup}

    # Nine slots, eleven players: the two lowest projections sit. Hal (2) still
    # starts -- he wins FLEX over Eli (1); it's Eli and the backup QB who bench.
    assert started == {"Ace Quarter", "Cal Runner", "Dex Runner", "Fay Wide",
                       "Gil Wide", "Ivy Tight", "Hal Wide", "KC DST", "Jay Kicker"}
    assert "Eli Runner" not in started and "Bench Quarter" not in started
    # Eli goes on to explode. The lineup was already locked; it must not change.
    actual = {**{k: 1.0 for k in proj}, "eli runner": 40.0}
    assert week_score(lineup, actual) == pytest.approx(9.0)   # 9 starters x 1.0


def test_every_slot_is_filled_with_an_eligible_player():
    proj = _proj(10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
    lineup = start_by_projection(ROSTER, proj, STARTERS)
    assert len(lineup) == len(STARTERS)
    assert [p["slot"] for p in lineup] == list(STARTERS)
    for p in lineup:
        if p["slot"] == "FLEX":
            assert p["position"] in {"RB", "WR", "TE"}
        elif p["slot"] not in ("FLEX",):
            assert p["position"] == p["slot"]
    # Nobody starts twice.
    assert len({p["player"] for p in lineup}) == len(STARTERS)


def test_an_unprojected_player_sits_when_anyone_else_can_play():
    """Injured/bye players simply aren't on the projection board. They must sort
    below everyone with a number rather than defaulting into the lineup."""
    proj = _proj(18, None, 12, 10, 9, 11, 9, 7, 6, 8, 7)   # backup QB absent
    started = {p["player"] for p in start_by_projection(ROSTER, proj, STARTERS)}
    assert "Bench Quarter" not in started and "Ace Quarter" in started


def test_waivers_only_move_when_the_STARTING_lineup_improves():
    """A better bench player is worth nothing if he never starts."""
    proj = _proj(20, 5, 15, 12, 1, 14, 11, 2, 8, 8, 7)
    # A free agent better than our worst bench WR but worse than the FLEX we
    # already start: he'd never crack the lineup, so no move.
    pool = [_p("Meh Wide", "WR")]
    proj["meh wide"] = 1.5
    same, move = run_waivers(ROSTER, pool, proj)
    assert move is None and same == ROSTER

    # Now one who would actually start.
    pool = [_p("Stud Wide", "WR")]
    proj["stud wide"] = 25
    new, move = run_waivers(ROSTER, pool, proj)
    assert move is not None and move["add"] == "Stud Wide" and move["gain"] > 0
    assert "Stud Wide" in {p["player"] for p in new}
    assert len(new) == len(ROSTER)


def test_waivers_never_pick_up_someone_already_rostered():
    proj = _proj(20, 5, 15, 12, 1, 99, 11, 2, 8, 8, 7)
    _, move = run_waivers(ROSTER, [_p("Fay Wide", "WR")], proj)
    assert move is None


def test_dst_naming_is_bridged_between_projection_and_result():
    """`project_kdst` says "PHI DST", `build_dst` says "PHI". Left unbridged a
    drafted defense scores zero all season and waivers churn the slot forever --
    which is exactly what happened on the first run of this simulation."""
    assert _dst_name("PHI", "DEF") == "PHI DST"
    assert _dst_name("PHI DST", "DEF") == "PHI DST"      # idempotent
    assert _dst_name("Harrison Butker", "K") == "Harrison Butker"


def test_roster_shape_matches_the_league():
    assert len(STARTERS) == 9
    assert ROSTER_SIZE == 14
    assert STARTERS.count("RB") == 2 and STARTERS.count("WR") == 2
    assert STARTERS.count("FLEX") == 1 and STARTERS.count("DEF") == 1
    # The caps must sum to exactly the roster, or the draft never reaches K/DST
    # at the bottom of the board and fields no defense at all.
    assert sum(LIMITS.values()) == ROSTER_SIZE


@requires_data_lake
def test_kdst_projection_and_result_names_actually_join():
    """Guards the silent-zero failure mode on real data: if these stop matching,
    every team's defense scores nothing and the simulation still 'works'."""
    from ffdata.db import connect
    from ffdata.kdst import project_kdst
    from ffdata.optimize import _norm
    from ffdata.scoring import STANDARD
    from ffdata.season_sim import _actual_points, _preseason_kdst

    con = connect()
    actual = set(_actual_points(con, 2024, STANDARD)["player"].map(_norm))
    pre = {_norm(p["player"]) for p in _preseason_kdst(con, 2024, STANDARD)
           if p["position"] == "DEF"}
    proj = {_norm(r["player_display_name"])
            for _, r in project_kdst(2024, 5, rules=STANDARD, con=con).iterrows()
            if r["position"] == "DEF"}
    assert len(pre) >= 32 and pre <= actual, "drafted defenses must be scoreable"
    assert len(proj) >= 32 and proj <= actual, "projected defenses must be scoreable"


@requires_data_lake
def test_the_draft_fields_a_legal_starting_lineup():
    """Every drafted roster must be able to fill all nine slots -- including the
    defense and kicker the season model doesn't rank."""
    import collections

    from ffdata.db import connect
    from ffdata.season_sim import draft_boards
    from ffdata.backtest_draft import run_snake_draft

    ours, naive = draft_boards(2024, con=connect())
    rosters = run_snake_draft([ours] + [naive] * 11, ROSTER_SIZE, LIMITS)
    for r in rosters:
        assert len(r) == ROSTER_SIZE
        got = collections.Counter(p["position"] for p in r)
        assert got["DEF"] == 1 and got["K"] == 1
        proj = {p["player"].lower().replace(" ", ""): 1.0 for p in r}
        assert len(start_by_projection(r, proj, STARTERS)) == len(STARTERS)


def test_playoff_bracket_byes_the_top_two_seeds():
    """A six-team bracket where every seed plays every round makes a title very
    nearly a coin flip: on 2024 that produced 1 title in 12 runs (the 1-in-12
    base rate) despite finishing first six times. Byes are what make the regular
    season worth playing."""
    import numpy as np

    scores = np.zeros((12, 3))
    seeds = [0, 1, 2, 3, 4, 5]
    scores[:, 0] = [999, 999, 10, 5, 4, 1, 0, 0, 0, 0, 0, 0]   # QF: 2>5, 3>4
    scores[:, 1] = [100, 100, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]    # SF: 1&2 seeds win
    scores[:, 2] = [100, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      # F: 1 seed wins
    champ, log = playoff_bracket(seeds, scores, [0, 1, 2])

    assert champ == 0
    qf = [g for g in log if g[0] == 0]
    assert len(qf) == 2, "only four teams play the quarterfinal"
    assert all(seeds[0] not in g[1:3] and seeds[1] not in g[1:3] for g in qf)
    # The 1 seed draws the WORSE survivor in the semi (here seed 3, not seed 2).
    assert (1, 0, 3, 0) in log


def test_playoff_bracket_breaks_ties_for_the_better_seed():
    import numpy as np

    scores = np.zeros((12, 3))
    seeds = [0, 1, 2, 3, 4, 5]
    scores[:, :] = 50.0            # everyone identical every round
    champ, _ = playoff_bracket(seeds, scores, [0, 1, 2])
    assert champ == 0, "dead-even scores must advance the better seed"
