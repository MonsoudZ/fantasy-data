"""Draft value logic: replacement levels, VOR-ranked availability, auction split."""

import pandas as pd

from conftest import requires_data_lake

from ffdata.draft import (_replacement_ranks, best_available, injury_context, keeper_value, trade_value,
                          round_cost, DEFAULT_LEAGUE)


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
    """A DuckDB with just the two views injury_context reads."""
    import duckdb
    con = duckdb.connect()
    con.register("injuries", pd.DataFrame(
        [(2025, *r) for r in rows],
        columns=["season", "gsis_id", "team", "week", "game_type",
                 "report_status", "report_primary_injury"]))
    con.register("rosters", pd.DataFrame(
        roster or [], columns=["season", "gsis_id", "status"]))
    return con


def test_injury_context_flags_who_limped_out_of_the_season():
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
    c = injury_context(2026, con=con).set_index("player_id")

    assert c.loc["late", "ended_hurt"]
    assert not c.loc["early", "ended_hurt"]
    # Distinct weeks: the duplicate week-21 report must not double-count.
    assert c.loc["late", "weeks_out"] == 2
    assert c.loc["late", "last_injury"] == "Knee" and c.loc["late", "last_round"] == "CON"
    # Never listed Out -> no injury note at all.
    assert "healthy" not in c.index or pd.isna(c.loc["healthy", "last_injury"])


def test_injury_context_ignores_absences_that_are_not_injuries():
    """The report doubles as an absence log. A personal matter isn't a health
    risk, and an illness resolves in days -- neither predicts Week 1 availability,
    though a missed game is still a missed game."""
    con = _inj_con([
        ("sick", "KC", 18, "REG", "Out", "Illness"),
        ("personal", "KC", 18, "REG", "Out", "Not injury related - personal matter"),
        ("hurt", "KC", 18, "REG", "Out", "Hamstring"),
    ])
    c = injury_context(2026, con=con).set_index("player_id")

    assert "personal" not in c.index                  # dropped outright
    assert not c.loc["sick", "ended_hurt"]            # counted, but not a flag
    assert c.loc["sick", "weeks_out"] == 1
    assert c.loc["hurt", "ended_hurt"]                # the real one still fires


def test_injury_context_reports_current_roster_status():
    """The freshest signal in July isn't last December -- it's a player still
    sitting on IR right now. It must surface even with no injury history."""
    con = _inj_con([], roster=[(2026, "ir", "RES"), (2026, "gone", "RET"),
                               (2026, "fine", "ACT")])
    c = injury_context(2026, con=con).set_index("player_id")

    assert c.loc["ir", "status"] == "on injured reserve"
    assert c.loc["gone", "status"] == "retired"
    assert "fine" not in c.index                      # ACT is not a note


def test_injury_context_survives_a_lake_without_injuries():
    import duckdb
    empty = injury_context(2026, con=duckdb.connect())
    assert empty.empty and "ended_hurt" in empty.columns
