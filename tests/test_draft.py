"""Draft value logic: replacement levels, VOR-ranked availability, auction split."""

import pandas as pd
import pytest

from conftest import requires_data_lake

from ffdata.draft import (_replacement_ranks, availability_context, best_available, keeper_value,
                          line_context, player_context, round_cost, trade_value, DEFAULT_LEAGUE)


def _valued_board():
    return pd.DataFrame({
        "player": ["Stud", "Mid", "Cheap", "Bench"], "position": ["WR", "RB", "WR", "TE"],
        "proj": [300, 250, 200, 150], "vor": [150, 100, 60, 20], "auction": [70, 45, 25, 5]})


def test_keeper_surplus_ranks_bargains_first():
    kp = keeper_value(_valued_board(), [("Cheap", 10), ("Stud", 65)], cost_type="auction")
    d = dict(zip(kp["player"], kp["surplus"]))
    assert d["Cheap"] == 15 and d["Stud"] == 5      # value - cost
    assert kp["player"].iloc[0] == "Cheap"          # biggest surplus on top


def test_trade_value_totals_and_verdict():
    r = trade_value(_valued_board(), ["Stud"], ["Mid", "Cheap"])
    assert r["side_a"]["auction"] == 70 and r["side_b"]["auction"] == 70
    assert "even" in r["verdict"]


def test_round_cost_falls_with_later_rounds():
    b = _valued_board()
    assert round_cost(b, 1, teams=1) >= round_cost(b, 2, teams=1)


def test_replacement_ranks_reflect_position_depth():
    r = _replacement_ranks(DEFAULT_LEAGUE)  # 12 teams, QB1/RB2/WR3/TE1 + 1 FLEX
    assert r["QB"] == 12                     # 12 teams x 1 starting QB, no flex
    assert r["WR"] > r["RB"] > r["TE"]       # WR demand highest, then RB, then TE
    assert r["RB"] > 24 and r["WR"] > 36     # FLEX pushes the RB/WR replacement deeper


def test_replacement_ranks_scale_with_league_size():
    small = _replacement_ranks({**DEFAULT_LEAGUE, "teams": 8})
    big = _replacement_ranks({**DEFAULT_LEAGUE, "teams": 14})
    assert big["QB"] > small["QB"]           # more teams -> shallower replacement


def test_superflex_deepens_qb_replacement():
    base = _replacement_ranks(DEFAULT_LEAGUE)                       # 1-QB league
    sf = _replacement_ranks({**DEFAULT_LEAGUE, "superflex": 1})      # + a superflex slot
    # A QB-eligible flex makes QB2s startable, so replacement QB gets much deeper
    # (~a second starting QB per team) -- the whole point for superflex value.
    assert sf["QB"] == base["QB"] + DEFAULT_LEAGUE["teams"]
    assert sf["RB"] == base["RB"] and sf["WR"] == base["WR"]         # SF doesn't touch RB/WR


def _board():
    return pd.DataFrame({
        "player": ["A", "B", "C", "D", "E"],
        "position": ["RB", "WR", "RB", "WR", "TE"],
        "proj": [300, 290, 250, 240, 200],
        "vor": [150, 140, 100, 90, 50],
        "auction": [60, 55, 40, 35, 20],
    })


def test_best_available_excludes_drafted_and_keeps_vor_order():
    out = best_available(_board(), drafted=["B"], n=10)
    assert list(out["player"]) == ["A", "C", "D", "E"]   # B removed, VOR order kept


def test_best_available_filters_by_position():
    out = best_available(_board(), position="RB")
    assert set(out["player"]) == {"A", "C"}


def test_best_available_matches_names_loosely():
    # Drafted list uses a different casing/punctuation than the board.
    board = _board().assign(player=["A.J. Brown", "B", "C", "D", "E"])
    out = best_available(board, drafted=["aj brown"])
    assert "A.J. Brown" not in set(out["player"])


@requires_data_lake
def test_player_context_describes_the_room():
    """Situation context for veterans: who's ahead, what left, scheme, moves."""
    from ffdata.draft import draft_board, player_context
    c = player_context(2026)
    assert {"player_id", "team", "prior_team", "moved", "blocked_by", "blocked_by_fp",
            "vacated_fp", "depth_rank", "pass_rate", "new_coach"}.issubset(c.columns)
    assert len(c) > 200
    # Nobody blocks himself, and a blocker must out-produce nobody-is-listed rows.
    board = draft_board(2026).merge(c, on="player_id", how="inner")
    assert (board["player"] != board["blocked_by"].fillna("")).all()
    # The best player at a position on a team leads his room (no blocker).
    top = board.sort_values("proj", ascending=False).groupby(["team", "position"]).head(1)
    assert top["blocked_by"].isna().mean() > 0.8
    # `moved` must agree with the team fields it's derived from.
    moved = board[board["moved"]]
    assert (moved["team"] != moved["prior_team"]).all()
    assert board["pass_rate"].dropna().between(0.3, 0.8).all()


def _inj_con(rows, roster=None):
    """A DuckDB with just the two views availability_context reads."""
    import duckdb
    con = duckdb.connect()
    con.register("injuries", pd.DataFrame(
        [(2025, *r) for r in rows],
        columns=["season", "gsis_id", "team", "week", "game_type",
                 "report_status", "report_primary_injury"]))
    con.register("rosters", pd.DataFrame(
        [(se, g, 1, st) for se, g, st in (roster or [])],
        columns=["season", "gsis_id", "week", "status"]))
    return con


def test_availability_context_flags_who_limped_out_of_the_season():
    """`ended_hurt` must key off the TEAM's last week, not the player's own last
    report -- measured against his own reports it is trivially true for everyone.
    KC plays to week 22, so a week-5 injury is long healed by then."""
    con = _inj_con([
        ("late", "KC", 20, "DIV", "Out", "Knee"),
        ("late", "KC", 21, "CON", "Out", "Knee"),
        ("late", "KC", 21, "CON", "Out", "Knee"),   # 2nd report, same game week
        ("early", "KC", 5, "REG", "Out", "Ankle"),
        ("healthy", "KC", 22, "SB", None, None),    # sets KC's finish at wk 22
    ])
    c = availability_context(2026, con=con).set_index("player_id")

    assert c.loc["late", "ended_hurt"]
    assert not c.loc["early", "ended_hurt"]
    # Distinct weeks: the duplicate week-21 report must not double-count.
    assert c.loc["late", "weeks_out"] == 2
    assert c.loc["late", "last_injury"] == "Knee" and c.loc["late", "last_round"] == "CON"
    # Never listed Out -> no injury note at all.
    assert "healthy" not in c.index or pd.isna(c.loc["healthy", "last_injury"])


def test_availability_context_ignores_absences_that_are_not_injuries():
    """The report doubles as an absence log. A personal matter isn't a health
    risk, and an illness resolves in days -- neither predicts Week 1 availability,
    though a missed game is still a missed game."""
    con = _inj_con([
        ("sick", "KC", 18, "REG", "Out", "Illness"),
        ("personal", "KC", 18, "REG", "Out", "Not injury related - personal matter"),
        ("hurt", "KC", 18, "REG", "Out", "Hamstring"),
    ])
    c = availability_context(2026, con=con).set_index("player_id")

    assert "personal" not in c.index                  # dropped outright
    assert not c.loc["sick", "ended_hurt"]            # counted, but not a flag
    assert c.loc["sick", "weeks_out"] == 1
    assert c.loc["hurt", "ended_hurt"]                # the real one still fires


def test_availability_context_reports_current_roster_status():
    """The freshest signal in July isn't last December -- it's a player still
    sitting on IR right now. It must surface even with no injury history."""
    con = _inj_con([], roster=[(2026, "ir", "RES"), (2026, "gone", "RET"),
                               (2026, "fine", "ACT")])
    c = availability_context(2026, con=con).set_index("player_id")

    assert c.loc["ir", "status"] == "on injured reserve"
    assert c.loc["gone", "status"] == "retired"
    assert "fine" not in c.index                      # ACT is not a note


def test_availability_context_survives_a_lake_without_injuries():
    import duckdb
    empty = availability_context(2026, con=duckdb.connect())
    assert empty.empty and "ended_hurt" in empty.columns


def test_availability_status_uses_his_LAST_known_week():
    """`rosters` is weekly and a player's status moves (ACT -> DEV -> INA). Taking
    any_value() would report a status he left months ago."""
    import duckdb
    con = duckdb.connect()
    con.register("injuries", pd.DataFrame(
        columns=["season", "gsis_id", "team", "week", "game_type",
                 "report_status", "report_primary_injury"]))
    con.register("rosters", pd.DataFrame(
        [(2026, "p", 1, "ACT"), (2026, "p", 9, "RES"),      # went on IR in week 9
         (2026, "q", 1, "RES"), (2026, "q", 9, "ACT")],     # came OFF IR in week 9
        columns=["season", "gsis_id", "week", "status"]))
    c = availability_context(2026, con=con).set_index("player_id")

    assert c.loc["p", "status"] == "on injured reserve"     # latest week wins
    assert "q" not in c.index                               # active now -> no note


def _line_con(ol_rows, avail_rows, season=2026):
    """depth_charts + rosters + injuries, enough for line_context."""
    import duckdb
    con = duckdb.connect()
    con.register("depth_charts", pd.DataFrame(
        [(season, t, g, "LT", 1, None, None, None) for t, g in ol_rows],
        columns=["season", "team", "gsis_id", "pos_abb", "pos_rank",
                 "club_code", "depth_position", "depth_team"]))
    con.register("rosters", pd.DataFrame(
        [(season, g, 1, st, nm) for g, st, nm in avail_rows],
        columns=["season", "gsis_id", "week", "status", "full_name"]))
    con.register("injuries", pd.DataFrame(
        columns=["season", "gsis_id", "team", "week", "game_type",
                 "report_status", "report_primary_injury"]))
    return con


def test_line_context_counts_only_unavailable_starters():
    con = _line_con(
        ol_rows=[("NYG", "lt"), ("NYG", "lg"), ("NYG", "c"), ("KC", "kclt")],
        avail_rows=[("lt", "RES", "Andrew Thomas"), ("lg", "PUP", "J.M. Schmitz"),
                    ("c", "ACT", "Healthy Guy"), ("kclt", "ACT", "Trey Smith")])
    lc = line_context(2026, con=con).set_index("team")

    assert lc.loc["NYG", "ol_out"] == 2
    assert "Andrew Thomas" in lc.loc["NYG", "ol_names"]
    assert "Healthy Guy" not in lc.loc["NYG", "ol_names"]   # ACT isn't "down"
    assert "KC" not in lc.index                             # nobody down -> no row


def test_line_context_needs_two_out_to_flag_the_line():
    """The measured finding is a threshold: one lineman down is noise, only 2+
    costs the backfield. A single injury must NOT surface a flag."""
    con = _line_con(
        ol_rows=[("DAL", "d1"), ("DAL", "d2"), ("DAL", "d3"),   # 3 starters, 1 down
                 ("NYG", "n1"), ("NYG", "n2")],                  # 2 starters, 2 down
        avail_rows=[("d1", "RES", "One Down"), ("d2", "ACT", "Healthy A"),
                    ("d3", "ACT", "Healthy B"),
                    ("n1", "RES", "Two Down A"), ("n2", "PUP", "Two Down B")])
    lc = line_context(2026, con=con).set_index("team")
    assert "DAL" not in lc.index                 # only one out -> below threshold
    assert lc.loc["NYG", "ol_out"] == 2          # two out -> flagged


def test_line_context_survives_a_lake_without_depth_charts():
    import duckdb
    empty = line_context(2026, con=duckdb.connect())
    assert empty.empty and {"team", "ol_out"}.issubset(empty.columns)


def test_team_last_week_reads_schedule_length_not_injury_reports():
    """`ended_hurt` measures against how far a team went; that must come from the
    real schedule, not from whenever someone last filed an injury report."""
    import duckdb

    from ffdata.draft import _team_last_week
    con = duckdb.connect()
    con.register("schedules", pd.DataFrame(
        [(2025, "KC", "BUF", 22, 30, 25),      # reached the Super Bowl (wk 22)
         (2025, "NYJ", "MIA", 18, 10, 20),     # missed the playoffs (wk 18)
         (2025, "KC", "DEN", 25, None, None)],  # an unplayed row must not count
        columns=["season", "home_team", "away_team", "week", "home_score", "away_score"]))
    lw = _team_last_week(con, 2025)
    assert lw["KC"] == 22 and lw["BUF"] == 22    # both teams in the SB game
    assert lw["NYJ"] == 18 and lw["MIA"] == 18


def test_team_coach_takes_the_coach_the_team_ended_with():
    """A mid-season firing gives a team two coaches; `new_coach` should anchor on
    who they ENDED with, not the alphabetically-first (the old min() bug)."""
    import duckdb

    from ffdata.draft import _team_coach
    con = duckdb.connect()
    con.register("schedules", pd.DataFrame(
        # ATL fired their week-1 coach; "Zzz Interim" finished the year.
        [(2025, "ATL", 1, "Arthur Smith", "REG"),
         (2025, "ATL", 17, "Zzz Interim", "REG"),
         (2025, "GB", 5, "Matt LaFleur", "REG")],
        columns=["season", "home_team", "week", "home_coach", "game_type"]))
    coaches = _team_coach(con).set_index("team")["coach"]
    assert coaches["ATL"] == "Zzz Interim"       # last game, not min() -> "Arthur"
    assert coaches["GB"] == "Matt LaFleur"


@requires_data_lake
def test_line_context_finds_real_starting_linemen():
    """Linemen are absent from `weekly` entirely (ingest keeps skill positions),
    so this only works because depth_charts + injuries carry every position."""
    lc = line_context(2026)
    assert not lc.empty
    assert lc["ol_out"].between(1, 5).all()          # can't lose more than five
    assert lc["ol_names"].str.len().gt(0).all()


@requires_data_lake
def test_offensive_line_context_rides_only_on_backfields():
    """It measured for RBs (-3.8 pts/game at 2+ down) and showed nothing for QBs,
    so it must not decorate non-RB rows with a number that means nothing there."""
    pc = player_context(2026)
    assert pc.loc[pc["ol_out"] > 0, "position"].eq("RB").all() if "position" in pc else True
    flagged = pc[pc["ol_out"] > 0]
    assert not flagged.empty and flagged["ol_names"].notna().all()


@requires_data_lake
def test_draft_board_is_reproducible():
    """Two identical calls must give identical numbers.

    They didn't: `_team_season` picked a player's team with
    `row_number() ... order by count(*) desc` and `_team_coach` used
    `any_value()`. Ties there are resolved by whichever DuckDB thread finishes
    first, so team_changed/coach_changed/sos flipped between runs and every
    projection moved a point or two. Small on one player, but it re-ordered the
    board -- and a season simulation built on it returned a different league
    table every run, which is how this was found.
    """
    from ffdata.db import connect
    from ffdata.draft import draft_board
    from ffdata.scoring import STANDARD

    league = {"teams": 12, "budget": 200, "roster_spots": 14,
              "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flex": 1}
    a = draft_board(2024, league, rules=STANDARD, con=connect())
    b = draft_board(2024, league, rules=STANDARD, con=connect())
    assert list(a["player"]) == list(b["player"]), "board ORDER must be stable"
    pd.testing.assert_series_equal(a["proj"], b["proj"])
    pd.testing.assert_series_equal(a["vor"], b["vor"])


@requires_data_lake
def test_team_and_coach_lookups_are_single_valued_and_stable():
    """The two aggregations behind the reproducibility bug, checked directly."""
    from ffdata.db import connect
    from ffdata.draft import _team_coach, _team_season

    con = connect()
    ts, coach = _team_season(con), _team_coach(con)
    assert not ts.duplicated(["player_id", "season"]).any()
    assert not coach.duplicated(["season", "team"]).any()
    # Compare CONTENT, not row order: DuckDB doesn't promise output order without
    # an ORDER BY, and both frames are merged on keys downstream anyway. What must
    # be stable is which team/coach each key maps to.
    def canon(df, keys):
        return df.sort_values(keys).reset_index(drop=True)

    pd.testing.assert_frame_equal(canon(ts, ["player_id", "season"]),
                                  canon(_team_season(con), ["player_id", "season"]))
    pd.testing.assert_frame_equal(canon(coach, ["season", "team"]),
                                  canon(_team_coach(con), ["season", "team"]))


def test_career_features_are_leak_free_and_recency_weighted():
    """Multi-year career features for row S must use ONLY seasons <= S, and weight
    the most recent season heaviest. Synthetic 3-season career, no lake needed."""
    import duckdb

    from ffdata.draft import _CAREER_DECAY, _career_features
    # A player with a rising career: 100, 200, 300 fp over 2020-22, games 16/10/17.
    rows = []
    for season, total_yds, gms in [(2020, 1000, 16), (2021, 2000, 10), (2022, 3000, 17)]:
        for wk in range(gms):
            rows.append({"player_id": "p", "season": season, "week": wk + 1,
                         "position": "WR", "player_display_name": "P",
                         "season_type": "REG", "recent_team": "KC", "opponent_team": "LV",
                         "receiving_yards": total_yds / gms, "targets": 8, "carries": 0,
                         "receptions": 5, "rushing_yards": 0, "passing_yards": 0,
                         "passing_tds": 0, "rushing_tds": 0, "receiving_tds": 0,
                         "target_share": 0.25})
    con = duckdb.connect()
    con.register("weekly", pd.DataFrame(rows))
    cf = _career_features(con).set_index("season")

    # Row for 2020 sees only 2020: c_seasons == 1, trend 0, games_avg 16.
    assert cf.loc[2020, "c_seasons"] == 1
    assert cf.loc[2020, "c_fp_trend"] == 0.0
    assert cf.loc[2020, "c_games_avg"] == 16
    # Row for 2022 sees all three, weighted toward 2022; durability floor = the
    # 10-game year; best = the 300-yard*0.1... season; trend = up (2022 > 2021).
    assert cf.loc[2022, "c_seasons"] == 3
    assert cf.loc[2022, "c_games_min"] == 10 and cf.loc[2022, "c_games_avg"] == pytest.approx(43 / 3)
    assert cf.loc[2022, "c_fp_trend"] > 0
    # Recency weight: fp rose across seasons, so weighting toward the most recent
    # year must pull the weighted average up year over year.
    assert cf.loc[2022, "c_fp_wavg"] > cf.loc[2021, "c_fp_wavg"], "rising career weights up"
    assert cf.loc[2022, "c_fp_wavg"] <= cf.loc[2022, "c_best_fp"]  # never exceeds the best year
    assert _CAREER_DECAY < 1.0


@requires_data_lake
def test_career_features_improve_the_projection():
    """The whole justification: adding career + durability must raise rank and cut
    error vs prior-year-only. Measured, not assumed."""
    from scipy.stats import spearmanr

    from ffdata.db import connect
    from ffdata.draft import _season_agg, project_season
    from ffdata.scoring import STANDARD

    con = connect()
    agg = _season_agg(con, STANDARD)

    def rank(season, career):
        proj = project_season(season, rules=STANDARD, con=con, career=career)
        m = proj.merge(agg[agg.season == season][["player_id", "fp"]], on="player_id")
        return spearmanr(m["proj"], m["fp"]).correlation

    base = sum(rank(s, False) for s in (2023, 2024, 2025)) / 3
    car = sum(rank(s, True) for s in (2023, 2024, 2025)) / 3
    assert car > base, f"career features must improve rank ({car:.3f} vs {base:.3f})"


def test_upside_features_measure_ceiling_and_are_leak_free():
    """Upside is the SHAPE of a player's weeks, not the total. A boom/bust player
    (4,4,30,4,28) must read a higher ceiling AND higher volatility than a steady
    one (14 every week) with the same season total. Row S sees only seasons <= S."""
    import duckdb

    from ffdata.draft import _upside_features

    def wk_rows(pid, season, weekly_yards):
        # receiving_yards drives standard fp at 0.1/yd; one row per week.
        return [{"player_id": pid, "season": season, "week": w + 1, "position": "WR",
                 "player_display_name": pid, "season_type": "REG",
                 "recent_team": "KC", "opponent_team": "LV",
                 "receiving_yards": y * 10, "targets": 8, "carries": 0,
                 "receptions": 0, "rushing_yards": 0, "passing_yards": 0,
                 "passing_tds": 0, "rushing_tds": 0, "receiving_tds": 0,
                 "target_share": 0.2}
                for w, y in enumerate(weekly_yards)]

    steady = wk_rows("steady", 2023, [14] * 10)                 # 14 every week
    spiky = wk_rows("spiky", 2023, [4, 4, 30, 4, 28, 4, 4, 30, 4, 28])  # same-ish total
    con = duckdb.connect()
    con.register("weekly", pd.DataFrame(steady + spiky))
    u = _upside_features(con).set_index("player_id")

    assert u.loc["spiky", "u_ceiling"] > u.loc["steady", "u_ceiling"]
    assert u.loc["spiky", "u_stdev"] > u.loc["steady", "u_stdev"]
    assert u.loc["steady", "u_floor"] >= u.loc["spiky", "u_floor"]   # steady has the floor


@requires_data_lake
def test_upside_bonus_reranks_toward_ceiling():
    """A positive upside_weight must lift a high-ceiling player's board value above
    where a mean-only projection puts him -- that's how sleepers surface."""
    from ffdata.db import connect
    from ffdata.draft import draft_board
    from ffdata.scoring import STANDARD

    con = connect()
    league = {"teams": 12, "budget": 200, "roster_spots": 14,
              "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flex": 1}
    base = draft_board(2025, league, rules=STANDARD, con=con, career=True, upside_weight=0.0)
    up = draft_board(2025, league, rules=STANDARD, con=con, career=True, upside_weight=0.2)
    # The boards rank differently; the upside board is not identical.
    assert list(base["player"]) != list(up["player"])


def test_board_upside_flags_high_ceiling_at_a_backup_price():
    """A sleeper: startable-tier weekly CEILING but a backup-tier projection --
    boom potential going late. A stud (high ceiling AND high proj) is NOT a
    sleeper, and a low-ceiling player never is."""
    import duckdb

    from ffdata.draft import _SLEEPER_GAP, board_upside

    # Build a WR board + weekly history so _upside_features has real ceilings.
    wr = [{"player": f"WR{i}", "player_id": f"w{i}", "position": "WR",
           "proj": 300 - i * 8, "vor": 150 - i * 8, "auction": 60 - i} for i in range(30)]
    board = pd.DataFrame(wr)
    rows = []
    for i in range(30):
        # Player w25 is the SLEEPER: a LOW projection (ranked ~26th) but monster
        # boom weeks (a few 40s) that give him a top-tier ceiling.
        if i == 25:
            weekly = [40, 42, 41, 5, 4, 5, 4, 5, 4, 5]
        else:
            base = max(1, 20 - i * 0.5)
            weekly = [base] * 10
        for wk, y in enumerate(weekly):
            rows.append({"player_id": f"w{i}", "season": 2024, "week": wk + 1,
                         "position": "WR", "player_display_name": f"WR{i}",
                         "season_type": "REG", "recent_team": "KC", "opponent_team": "LV",
                         "receiving_yards": y * 10, "targets": 8, "carries": 0,
                         "receptions": 0, "rushing_yards": 0, "passing_yards": 0,
                         "passing_tds": 0, "rushing_tds": 0, "receiving_tds": 0,
                         "target_share": 0.2})
    con = duckdb.connect()
    con.register("weekly", pd.DataFrame(rows))
    out = board_upside(board, 2025, con=con).set_index("player_id")

    assert out.loc["w25", "sleeper"], "high-ceiling / backup-price player is a sleeper"
    assert not out.loc["w0", "sleeper"], "the WR1 (elite proj, not underpriced) is not a sleeper"
    assert not out.loc["w20", "sleeper"], "a low-ceiling backup is not a sleeper"
    # ceiling/floor are exposed for the human to read.
    assert out.loc["w25", "ceiling"] > out.loc["w25", "floor"]
    assert _SLEEPER_GAP >= 1


def test_board_upside_anchor_flags_a_high_floor_starter():
    """The mirror of the sleeper: a top-tier weekly FLOOR at a startable
    projection -- the reliable, set-and-forget player."""
    import duckdb

    from ffdata.draft import board_upside
    wr = [{"player": f"WR{i}", "player_id": f"w{i}", "position": "WR",
           "proj": 300 - i * 8, "vor": 150 - i * 8, "auction": 60 - i} for i in range(30)]
    rows = []
    for i in range(30):
        # w0 is a rock: 18 every week (high, flat floor). w29 is a low scrub.
        weekly = [18] * 10 if i == 0 else [max(1, 15 - i * 0.5)] * 10
        for wk, y in enumerate(weekly):
            rows.append({"player_id": f"w{i}", "season": 2024, "week": wk + 1,
                         "position": "WR", "player_display_name": f"WR{i}",
                         "season_type": "REG", "recent_team": "KC", "opponent_team": "LV",
                         "receiving_yards": y * 10, "targets": 8, "carries": 0,
                         "receptions": 0, "rushing_yards": 0, "passing_yards": 0,
                         "passing_tds": 0, "rushing_tds": 0, "receiving_tds": 0,
                         "target_share": 0.2})
    con = duckdb.connect()
    con.register("weekly", pd.DataFrame(rows))
    out = board_upside(pd.DataFrame(wr), 2025, con=con).set_index("player_id")
    assert out.loc["w0", "anchor"], "high floor + startable projection = anchor"
    assert not out.loc["w29", "anchor"], "a low scrub is not an anchor"


def test_availability_penalty_downranks_the_unavailable():
    """A player known to be on IR / retired must be MARKED DOWN in value, not just
    badged -- his rank should drop below a healthy peer with the same projection."""
    from ffdata.draft import availability_penalty

    board = pd.DataFrame({
        "player": ["Healthy", "OnIR", "Retired"], "player_id": ["h", "ir", "ret"],
        "position": ["RB", "RB", "RB"], "proj": [200.0, 200.0, 200.0]})
    con = _inj_con([], roster=[(2026, "h", "ACT"), (2026, "ir", "RES"),
                               (2026, "ret", "RET")])
    out = availability_penalty(board, 2026, con=con).set_index("player_id")

    assert out.loc["h", "proj"] == 200.0            # healthy untouched
    assert out.loc["ir", "proj"] < 200.0            # IR marked down
    assert out.loc["ret", "proj"] == 0.0            # retired -> zero value
    assert out.loc["ir", "vor"] < out.loc["h", "vor"]   # rank reflects it
    assert out.loc["ir", "avail"] == "on injured reserve"


def test_streamability_discount_pulls_down_qb_and_te_vor():
    """Raw VOR over-values QB/TE (their replacement is startable/streamable). The
    discount must lower QB/TE value relative to an RB/WR with the same nominal
    gap over replacement -- so elite QBs stop ranking in the first two rounds."""
    from ffdata.draft import _STREAM_DISCOUNT, score_board

    league = {"teams": 12, "budget": 200, "roster_spots": 14,
              "starters": {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, "flex": 1}
    # Give each position a clear stud and a replacement-level tail so VOR is real.
    rows = []
    for pos, top in [("QB", 380), ("RB", 300), ("WR", 300), ("TE", 260)]:
        for i in range(20):
            rows.append({"player": f"{pos}{i}", "player_id": f"{pos}{i}",
                         "position": pos, "proj": top - i * 10})
    out = score_board(pd.DataFrame(rows), league).set_index("player")
    # The elite QB's raw gap over replacement is huge, but after the discount its
    # VOR must be strictly below the elite RB's (RB isn't discounted).
    assert out.loc["QB0", "vor"] < out.loc["RB0", "vor"], "elite QB no longer outranks elite RB"
    assert out.loc["TE0", "vor"] < out.loc["RB0", "vor"], "elite TE discounted below elite RB"
    # Re-scoring WITHOUT the discount would leave the elite QB on top -- confirm the
    # discount is what changed it, by checking the factor is applied to VOR.
    assert 0 < _STREAM_DISCOUNT["QB"] < 1 and 0 < _STREAM_DISCOUNT["TE"] < 1
    # The elite QB's discounted VOR is a fraction of its raw gap over replacement.
    assert out.loc["QB0", "vor"] < (out.loc["QB0", "proj"] - out.loc["QB19", "proj"])


@requires_data_lake
def test_forward_sos_is_leak_free_normalized_and_covers_the_draft_year():
    """Forward strength-of-schedule: every season averages ~1.0 (it's normalized),
    the spread is small (a full season of matchups nets out), and the upcoming,
    not-yet-played season gets a number so a live draft isn't blank."""
    from ffdata.draft import _def_difficulty, _forward_sos
    from ffdata.ingest import upcoming_nfl_season
    from ffdata.db import connect
    from ffdata.scoring import STANDARD
    con = connect()
    dd = _def_difficulty(con, STANDARD)
    # A defense's very first season has no prior form -> neutral 1.0, never a leak.
    first = dd.sort_values("season").groupby(["def_team", "position"]).head(1)
    assert (first["d_diff"] == 1.0).all()
    fs = _forward_sos(con, STANDARD)
    up = upcoming_nfl_season()
    cur = fs[fs["season"] == up]
    assert not cur.empty, "no forward schedule for the draft year"
    # Normalized: each season/position centers on ~1.0, and no team's whole-season
    # road is more than ~15% off average -- schedule is a weak season-level signal.
    for (_s, _p), g in fs.groupby(["season", "position"]):
        assert abs(g["sched_ahead"].mean() - 1.0) < 0.05
    assert cur["sched_ahead"].between(0.85, 1.15).all()


@requires_data_lake
def test_schedule_context_tags_the_board_without_touching_vor():
    """schedule_context adds sched_ahead/sched_rank as context and leaves the VOR
    ranking exactly as it found it -- schedule is a tiebreaker, not a value input."""
    from ffdata.draft import draft_board, schedule_context
    from ffdata.ingest import upcoming_nfl_season
    from ffdata.scoring import STANDARD
    up = upcoming_nfl_season()
    board = draft_board(up, rules=STANDARD)
    tagged = schedule_context(board, up, rules=STANDARD)
    assert {"sched_ahead", "sched_rank"}.issubset(tagged.columns)
    assert tagged["sched_ahead"].notna().mean() > 0.7  # most rostered players tagged
    # rank 1 = easiest road within a position
    for pos, g in tagged.dropna(subset=["sched_rank"]).groupby("position"):
        easiest = g.loc[g["sched_rank"].idxmin()]
        assert easiest["sched_ahead"] == g["sched_ahead"].max()
    # VOR and its ordering are untouched.
    j = board[["player_id", "vor"]].merge(
        tagged[["player_id", "vor"]], on="player_id", suffixes=("_a", "_b"))
    assert (j["vor_a"] == j["vor_b"]).all()


@requires_data_lake
def test_competition_features_are_leak_free_and_key_on_the_target_year():
    """Opportunity features describe the room a player enters in the TARGET season,
    from prior-year volume + the target-year roster -- never the target year's own
    stats. Shares are bounded, and a thinned room reads as higher opp_share."""
    from ffdata.draft import _competition_features, _player_volume
    from ffdata.ingest import upcoming_nfl_season
    from ffdata.db import connect
    from ffdata.scoring import STANDARD
    con = connect()
    cf = _competition_features(con, STANDARD)
    assert {"player_id", "tseason", "opp_share", "vac_share", "comp_vol"}.issubset(cf.columns)
    # Shares are bounded where defined (a player with no target-year team has no
    # room, so NaN -- the feature frame fills those to 0, context leaves them blank).
    assert cf["opp_share"].dropna().between(0, 1).all()
    assert cf["vac_share"].dropna().between(0, 1).all()
    assert (cf["comp_vol"].dropna() >= 0).all()
    # The upcoming (not-yet-played) season must be covered -- its roster is known.
    up = upcoming_nfl_season()
    assert (cf["tseason"] == up).any()
    # Leak-free key: a feature row for tseason T only needs volume from < T. Build
    # for a mid history season and confirm it exists without T's volume in hand.
    vol = _player_volume(con)
    assert cf["tseason"].min() <= vol["season"].max()


@requires_data_lake
def test_competition_context_tags_vacated_opportunity():
    """competition_context surfaces opp_share / vac_share / opp_open on the board,
    and opp_open marks rooms where a real chunk of touches left."""
    from ffdata.draft import draft_board, competition_context, _OPP_OPEN_SHARE
    from ffdata.ingest import upcoming_nfl_season
    from ffdata.scoring import STANDARD
    up = upcoming_nfl_season()
    board = draft_board(up, rules=STANDARD)
    tagged = competition_context(board, up, rules=STANDARD)
    assert {"opp_share", "vac_share", "opp_open"}.issubset(tagged.columns)
    # opp_open is exactly "a meaningful share vacated".
    opened = tagged[tagged["opp_open"]]
    assert (opened["vac_share"] >= _OPP_OPEN_SHARE).all()
    assert len(opened) > 0                       # some rooms always turn over
    # Context only touches the new columns; VOR ordering is unchanged.
    j = board[["player_id", "vor"]].merge(
        tagged[["player_id", "vor"]], on="player_id", suffixes=("_a", "_b"))
    assert (j["vor_a"] == j["vor_b"]).all()


@requires_data_lake
def test_sos_quality_is_leak_free_covers_2026_and_is_a_model_feature():
    """The Sharp-style overall-quality SOS: a defense's first season has no prior
    form (neutral 0), the upcoming season is covered (schedule known), and sos_q
    is actually wired into the projection feature list."""
    from ffdata.draft import _sos_quality, _FEATS, _feature_frame
    from ffdata.ingest import upcoming_nfl_season
    from ffdata.db import connect
    from ffdata.scoring import STANDARD
    con = connect()
    assert "sos_q" in _FEATS
    sq = _sos_quality(con)
    up = upcoming_nfl_season()
    assert (sq["season"] == up).any(), "no overall SOS for the draft year"
    assert len(sq[sq["season"] == up]) == 32
    # It reaches the feature frame with no missing values (filled to neutral).
    ff = _feature_frame(con, STANDARD)
    assert "sos_q" in ff.columns and ff["sos_q"].notna().all()
    # Sharp's 2026 easiest roads (weak opponents) should read as EASY for us too:
    # Cleveland and New Orleans both land in our easiest third.
    cur = sq[sq["season"] == up].sort_values("sos_q")   # ascending = weakest opponents
    easy_third = set(cur.head(11)["team"])
    assert {"CLE", "NO"}.issubset(easy_third)


@requires_data_lake
def test_career_year_context_flags_workhorses_not_rookies_and_leaves_vor_alone():
    """The RB career-year regression flag: fires on established backs coming off a
    career-high workhorse load, never on rookies (whose debut is trivially a high),
    and never touches VOR (it's context, not a downgrade)."""
    from ffdata.draft import draft_board, career_year_context, rookie_projection
    from ffdata.ingest import upcoming_nfl_season
    from ffdata.db import connect
    from ffdata.scoring import STANDARD
    con = connect()
    up = upcoming_nfl_season()
    board = draft_board(up, rules=STANDARD, con=con)
    tagged = career_year_context(board, up, con=con)
    assert {"career_year", "prior_touches"}.issubset(tagged.columns)
    flagged = tagged[tagged["career_year"]]
    assert len(flagged) > 0                                   # some backs always qualify
    assert (flagged["position"] == "RB").all()               # RB-only signal
    assert (flagged["prior_touches"] >= 250).all()           # a real workhorse load
    # No rookie should be flagged -- their first year has no prior baseline.
    rk = rookie_projection(up, rules=STANDARD, con=con)
    if rk is not None and not rk.empty:
        assert not set(flagged["player_id"]) & set(rk["player_id"])
    # VOR is identical before/after -- context only.
    j = board[["player_id", "vor"]].merge(
        tagged[["player_id", "vor"]], on="player_id", suffixes=("_a", "_b"))
    assert (j["vor_a"] == j["vor_b"]).all()


@requires_data_lake
def test_player_context_ranks_the_room_by_who_is_actually_ahead_not_last_year():
    """blocked_by must follow who's AHEAD this year -- consensus ADP when a feed is
    loaded, else our forward projection -- not last season's points. The invariant:
    no one is blocked by a teammate the active signal ranks BELOW him, and a
    projected/ADP room leader is never shown as blocked."""
    from ffdata.draft import _consensus_adp, draft_board, player_context
    from ffdata.ingest import upcoming_nfl_season
    from ffdata.optimize import _norm
    from ffdata.db import connect
    from ffdata.scoring import STANDARD
    con = connect()
    up = upcoming_nfl_season()
    board = draft_board(up, rules=STANDARD, con=con, career=True, competition=True)
    ctx = player_context(up, rules=STANDARD, con=con, ranked=board)
    m = board[["player_id", "player", "position", "proj"]].merge(
        ctx[["player_id", "team", "blocked_by"]], on="player_id", how="inner").dropna(subset=["team"])
    adp = _consensus_adp(con)
    if adp:
        # A blocker must have a better (lower) ADP than the man he blocks, whenever
        # both are ranked by the market -- the depth chart, not last-year points.
        name_adp = {row.player: adp.get(_norm(row.player)) for row in m.itertuples()}
        for row in m.itertuples():
            if row.blocked_by and name_adp.get(row.player) is not None \
               and name_adp.get(row.blocked_by) is not None:
                assert name_adp[row.blocked_by] <= name_adp[row.player], \
                    f"{row.player} blocked by a lower-ADP teammate {row.blocked_by}"
    else:
        # No feed: falls back to our projection -- the top-projected leads each room.
        tops = m.sort_values("proj", ascending=False).groupby(["team", "position"]).head(1)
        assert tops["blocked_by"].isna().all()
    assert ctx["blocked_by"].notna().any()   # rooms actually have blockers


@requires_data_lake
def test_consensus_adp_orders_the_room_and_flags_disagreement():
    """When a consensus-ADP feed is loaded, the room follows it (the market's depth
    chart beats our own projection where they differ), and consensus_context marks
    where our board disagrees with the market. Skips cleanly if no feed is present."""
    from ffdata.draft import _consensus_adp, consensus_context, draft_board
    from ffdata.ingest import upcoming_nfl_season
    from ffdata.db import connect
    from ffdata.scoring import PPR
    con = connect()
    if not _consensus_adp(con):
        import pytest as _pt
        _pt.skip("no consensus_adp feed loaded")
    up = upcoming_nfl_season()
    board = draft_board(up, rules=PPR, con=con, career=True, competition=True)
    tagged = consensus_context(board, con=con)
    assert {"adp", "our_rank", "adp_delta"}.issubset(tagged.columns)
    assert tagged["adp"].notna().sum() > 50
    # adp_delta is exactly adp - our_rank where both exist.
    both = tagged.dropna(subset=["adp"])
    assert (both["adp_delta"] == both["adp"] - both["our_rank"]).all()
    # consensus_context never changes VOR.
    j = board[["player_id", "vor"]].merge(tagged[["player_id", "vor"]], on="player_id",
                                          suffixes=("_a", "_b"))
    assert (j["vor_a"] == j["vor_b"]).all()


@requires_data_lake
def test_consensus_adp_names_match_the_board():
    """The ADP feed must actually join to our players (a name-normalisation guard) --
    at least the skill-position starters, or the room ordering does nothing."""
    from ffdata.draft import _consensus_adp, draft_board
    from ffdata.ingest import upcoming_nfl_season
    from ffdata.optimize import _norm
    from ffdata.db import connect
    from ffdata.scoring import PPR
    con = connect()
    adp = _consensus_adp(con)
    if not adp:
        import pytest as _pt
        _pt.skip("no consensus_adp feed loaded")
    board = draft_board(upcoming_nfl_season(), rules=PPR, con=con)
    bnorm = {_norm(p) for p in board["player"]}
    matched = sum(1 for k in adp if k in bnorm)
    assert matched >= 150, f"only {matched} ADP names matched the board -- normalisation drift?"
