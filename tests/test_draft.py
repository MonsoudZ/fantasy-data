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
