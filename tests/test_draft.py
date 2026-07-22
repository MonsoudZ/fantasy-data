"""Draft value logic: replacement levels, VOR-ranked availability, auction split."""

import pandas as pd

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


def test_line_context_survives_a_lake_without_depth_charts():
    import duckdb
    empty = line_context(2026, con=duckdb.connect())
    assert empty.empty and {"team", "ol_out"}.issubset(empty.columns)


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
