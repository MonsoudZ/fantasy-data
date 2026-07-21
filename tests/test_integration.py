"""End-to-end checks against the real data lake. Skipped when none is present.

Run `python -m ffdata.cli` first to enable these locally; CI skips them (no
committed data) and relies on the synthetic unit tests.
"""

import pandas as pd

from ffdata.db import connect
from ffdata.scoring import score, PPR, STANDARD
from conftest import requires_data_lake


@requires_data_lake
def test_score_matches_nflverse_ppr_exactly():
    con = connect()
    latest = con.sql("select max(season) from weekly").fetchone()[0]
    w = con.sql(f"select * from weekly where season = {latest}").df()
    scored = score(w, PPR)
    ref = w["fantasy_points_ppr"].fillna(0).round(2)
    assert (scored["fp"] - ref).abs().max() == 0.0


@requires_data_lake
def test_score_matches_nflverse_standard_exactly():
    con = connect()
    latest = con.sql("select max(season) from weekly").fetchone()[0]
    w = con.sql(f"select * from weekly where season = {latest}").df()
    scored = score(w, STANDARD, col="fp_std")
    ref = w["fantasy_points"].fillna(0).round(2)
    assert (scored["fp_std"] - ref).abs().max() == 0.0


@requires_data_lake
def test_views_exist_and_are_queryable():
    con = connect()
    tables = {r[0] for r in con.sql("show tables").fetchall()}
    assert "weekly" in tables and "schedules" in tables
    assert con.sql("select count(*) from weekly").fetchone()[0] > 0
