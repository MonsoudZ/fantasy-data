"""Kicker + team-defense scoring and leak-free trailing projection (synthetic).

No data lake here, so these exercise the scoring math and the trailing logic on
hand-built frames -- the same discipline as the other unit tests. The projection
*magnitudes* against real nflverse data are validated separately (see kdst.py's
module docstring); this proves the arithmetic and the leak-free window.
"""

import pandas as pd

from conftest import requires_data_lake
from ffdata.kdst import (
    _dst_pa_points, _trailing_pred, project_kdst, score_dst, score_kicker,
)
from ffdata.scoring import PPR, ScoringRules


def test_kicker_scoring_uses_distance_buckets_when_present():
    df = pd.DataFrame({"fg_made_0_19": [1], "fg_made_30_39": [1], "fg_made_40_49": [1],
                       "fg_made_50_59": [1], "pat_made": [3], "fg_missed": [1]})
    # 3 (short) + 3 (short) + 4 (mid) + 5 (long) + 3*1 (pat) + 0 (miss default) = 18
    assert score_kicker(df, PPR)["fp"].iloc[0] == 18.0
    # A league that penalizes misses subtracts a point.
    assert score_kicker(df, ScoringRules(fg_miss=-1.0))["fp"].iloc[0] == 17.0


def test_kicker_scoring_falls_back_to_flat_made_fgs():
    df = pd.DataFrame({"fg_made": [2], "pat_made": [1]})   # no distance columns
    assert score_kicker(df, PPR)["fp"].iloc[0] == 2 * 3 + 1   # 7


def test_dst_points_allowed_tiers():
    assert [_dst_pa_points(x) for x in (0, 3, 10, 17, 24, 30, 45)] == \
        [10.0, 7.0, 4.0, 1.0, 0.0, -1.0, -4.0]


def test_dst_scoring_combines_counting_stats_and_points_allowed():
    df = pd.DataFrame({"def_sacks": [3], "def_interceptions": [2], "def_tds": [1],
                       "points_allowed": [10]})
    # 3*1 + 2*2 + 1*6 + tier(10)=4  =  17
    assert score_dst(df, PPR)["fp"].iloc[0] == 17.0


def test_trailing_pred_is_leak_free():
    scored = pd.DataFrame({"team": ["BUF"] * 4, "season": [2024] * 4,
                           "week": [1, 2, 3, 4], "fp": [10, 20, 30, 40]})
    # As of week 4, only weeks 1-3 count -> mean(10,20,30) = 20; week 40 excluded.
    assert _trailing_pred(scored, "team", 2024, 4, 5)["pred"].iloc[0] == 20.0
    # Week 1 has no prior games -> no projection.
    assert _trailing_pred(scored, "team", 2024, 1, 5).empty


def test_trailing_pred_window_caps_history():
    scored = pd.DataFrame({"team": ["BUF"] * 4, "season": [2024] * 4,
                           "week": [1, 2, 3, 4], "fp": [0, 0, 30, 30]})
    # window=2 keeps only weeks 2,3 before week 4 -> mean(0,30) = 15.
    assert _trailing_pred(scored, "team", 2024, 4, window=2)["pred"].iloc[0] == 15.0


def test_project_kdst_degrades_to_empty_without_a_lake(monkeypatch):
    """Simulate a missing lake explicitly -- don't rely on the ambient env, or
    this passes in CI and fails on any machine that has ingested data."""
    import ffdata.kdst as kdst

    def _no_lake():
        raise FileNotFoundError("no data lake")
    monkeypatch.setattr(kdst, "connect", _no_lake)
    board = project_kdst(2024, 5)
    assert board.empty
    assert list(board.columns) == ["player_display_name", "position", "pred", "recent_team"]


@requires_data_lake
def test_project_kdst_projects_real_kickers_and_defenses():
    """Against the real lake: both K and DEF must appear. Guards two bugs the
    synthetic tests missed -- kickers being filtered out of `weekly` at ingest,
    and retired players keeping a trailing average forever."""
    board = project_kdst(2024, 5)
    assert not board.empty
    by_pos = board["position"].value_counts()
    assert by_pos.get("K", 0) > 0, "no kickers -- are K rows being dropped at ingest?"
    assert by_pos.get("DEF", 0) > 0
    # A DST can legitimately project negative (points allowed, no takeaways);
    # kickers shouldn't. Everything should be finite and in a sane range.
    assert board["pred"].notna().all()
    assert board["pred"].between(-20, 40).all()
    assert (board.loc[board["position"] == "K", "pred"] >= 0).all()
    # Nobody long-retired: Adam Vinatieri's last season was 2019.
    assert "Adam Vinatieri" not in set(board["player_display_name"])


def test_project_kdst_assembles_board_rows(monkeypatch):
    import ffdata.kdst as kdst

    kick = pd.DataFrame({
        "player_display_name": ["Justin Tucker"] * 3,
        "recent_team": ["BAL"] * 3, "position": ["K"] * 3,
        "season": [2024] * 3, "week": [1, 2, 3], "fg_made": [2, 3, 1], "pat_made": [1, 2, 3],
    })
    dst = pd.DataFrame({
        "team": ["BUF"] * 3, "position": ["DEF"] * 3,
        "season": [2024] * 3, "week": [1, 2, 3], "points_allowed": [3, 10, 45],
    })
    monkeypatch.setattr(kdst, "build_kicker", lambda con=None: kick)
    monkeypatch.setattr(kdst, "build_dst", lambda con=None: dst)

    board = kdst.project_kdst(2024, 4, con=object())   # con is unused past the builders
    rows = {r["player_display_name"]: r for _, r in board.iterrows()}
    # Kicker: trailing mean of scored weeks 1-3. fp = 3*fg_made + pat -> 7,11,6 -> mean 8.0.
    assert rows["Justin Tucker"]["position"] == "K"
    assert rows["Justin Tucker"]["pred"] == 8.0 and rows["Justin Tucker"]["recent_team"] == "BAL"
    # Defense named "<TEAM> DST"; PA tiers 7,4,-4 -> mean -> 2.33.
    assert rows["BUF DST"]["position"] == "DEF" and rows["BUF DST"]["recent_team"] == "BUF"
    assert round(rows["BUF DST"]["pred"], 2) == 2.33
