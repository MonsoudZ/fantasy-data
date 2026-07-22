"""Kicker (K) and team-defense (DST) scoring + trailing projections.

Standard leagues start a K and a DEF (QB/RB/RB/WR/WR/TE/FLEX/DEF/K), so the app
scores and projects them too. These are the least predictable positions in
fantasy -- their weekly outcome is dominated by game script and touchback luck --
so the honest projection is a trailing average, not a fancy model. That's the
same irreducible-floor lesson the skill positions taught (confirmed six ways),
only more so here.

    from ffdata.kdst import project_kdst
    project_kdst(2024, 15)      # K + DEF board rows: player_display_name/position/pred

⚠️  HONESTY / VALIDATION. The scoring math and the leak-free trailing logic are
unit-tested on synthetic frames. The *magnitudes* are UNVALIDATED against real
data -- there is no data lake (or network to nflverse) in the environment this
was built in. Two specific gaps to check once you've ingested:
  * Kicker distance buckets: nflverse column names vary by schema era; this reads
    the documented `fg_made_0_19 / _20_29 / _30_39 / _40_49 / _50_59 / _60_`
    splits when present and falls back to a flat `fg_made` value otherwise.
  * DST counting stats (sacks, takeaways, defensive TDs) need a defensive stats
    source this project doesn't ingest yet; points-allowed comes straight from
    `schedules` (reliable), and the rest default to zero. So a DST projection
    here is points-allowed-dominated -- honest but incomplete. Wire a defensive
    box-score source and backtest before trusting the numbers.
"""

from __future__ import annotations

import pandas as pd

from .db import connect
from .scoring import PPR, ScoringRules

FORM_WINDOW = 5   # trailing games for the K/DST average
_COLS = ["player_display_name", "position", "pred", "recent_team"]


def _g(df: pd.DataFrame, c: str) -> pd.Series:
    """Column or an all-zero series (nflverse schemas vary; missing stat == 0)."""
    return df.get(c, pd.Series(0.0, index=df.index)).fillna(0.0)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_kicker(df: pd.DataFrame, rules: ScoringRules = PPR, col: str = "fp") -> pd.DataFrame:
    """Fantasy points for kicker rows. Uses distance buckets when the columns are
    present (the usual 3/4/5-pt ladder); otherwise a flat made-FG value. Extra
    points and (optionally penalized) misses always apply."""
    if any(c in df.columns for c in ("fg_made_0_19", "fg_made_40_49", "fg_made_50_59")):
        short = _g(df, "fg_made_0_19") + _g(df, "fg_made_20_29") + _g(df, "fg_made_30_39")
        mid = _g(df, "fg_made_40_49")
        lng = _g(df, "fg_made_50_59") + _g(df, "fg_made_60_") + _g(df, "fg_made_60_plus")
        fg = short * rules.fg_0_39 + mid * rules.fg_40_49 + lng * rules.fg_50_plus
    else:
        fg = _g(df, "fg_made") * rules.fg_0_39
    pts = fg + _g(df, "pat_made") * rules.pat + _g(df, "fg_missed") * rules.fg_miss
    out = df.copy()
    out[col] = pts.round(2)
    return out


# (max points allowed, points scored) -- the standard mainstream ladder; 35+ is -4.
_PA_TIERS = [(0, 10.0), (6, 7.0), (13, 4.0), (20, 1.0), (27, 0.0), (34, -1.0)]


def _dst_pa_points(pa: float) -> float:
    for hi, pts in _PA_TIERS:
        if pa <= hi:
            return pts
    return -4.0


def score_dst(df: pd.DataFrame, rules: ScoringRules = PPR, col: str = "fp") -> pd.DataFrame:
    """Fantasy points for team-defense rows: counting stats (where present) plus
    the points-allowed tier bonus (the dominant, always-available term)."""
    pts = (_g(df, "def_sacks") * rules.dst_sack
           + _g(df, "def_interceptions") * rules.dst_int
           + _g(df, "def_fumbles_recovered") * rules.dst_fumble_rec
           + _g(df, "def_tds") * rules.dst_td
           + _g(df, "def_safeties") * rules.dst_safety
           + _g(df, "def_blocks") * rules.dst_block)
    if "points_allowed" in df.columns:
        pts = pts + df["points_allowed"].fillna(0).map(_dst_pa_points)
    out = df.copy()
    out[col] = pts.round(2)
    return out


# --------------------------------------------------------------------------- #
# Raw tables (from the lake) -> scored weekly frames
# --------------------------------------------------------------------------- #

def build_kicker(con=None) -> pd.DataFrame:
    """Kicker weeks from the `weekly` table (position K), or empty if unavailable."""
    con = con or connect()
    try:
        return con.sql("select * from weekly where position = 'K'").df()
    except Exception:  # noqa: BLE001 - no lake / no kicker rows -> degrade to empty
        return pd.DataFrame()


def build_dst(con=None) -> pd.DataFrame:
    """One row per team per game with points allowed (from `schedules`). Defensive
    counting stats aren't sourced yet, so only points_allowed is populated."""
    con = con or connect()
    try:
        g = con.sql("""
            select season, week, home_team, away_team, home_score, away_score
            from schedules where home_score is not null and week is not null
        """).df()
    except Exception:  # noqa: BLE001 - no lake -> degrade to empty
        return pd.DataFrame()
    home = g.rename(columns={"home_team": "team", "away_score": "points_allowed"})
    away = g.rename(columns={"away_team": "team", "home_score": "points_allowed"})
    keep = ["season", "week", "team", "points_allowed"]
    d = pd.concat([home[keep], away[keep]], ignore_index=True)
    d["position"] = "DEF"
    return d


# --------------------------------------------------------------------------- #
# Leak-free trailing projection
# --------------------------------------------------------------------------- #

def _trailing_pred(scored: pd.DataFrame, key: str, season: int, week: int,
                   window: int) -> pd.DataFrame:
    """Per-entity mean fantasy points over the last `window` games STRICTLY BEFORE
    (season, week) -- the leak-free 'form' as of that week. Returns key -> pred."""
    s = scored.assign(_k=scored["season"] * 100 + scored["week"])
    prior = s[s["_k"] < season * 100 + week].sort_values([key, "_k"])
    if prior.empty:
        return pd.DataFrame(columns=[key, "pred"])
    pred = prior.groupby(key).tail(window).groupby(key)["fp"].mean().round(2)
    return pred.reset_index().rename(columns={"fp": "pred"})


def project_kdst(season: int, week: int, rules: ScoringRules = PPR, con=None,
                 window: int = FORM_WINDOW) -> pd.DataFrame:
    """K + DEF projection board rows for (season, week): trailing-average fantasy
    points per kicker and per team defense. Leak-free (only prior weeks feed the
    average). Returns an empty board (no error) when the lake isn't available."""
    if con is None:
        try:
            con = connect()
        except Exception:  # noqa: BLE001 - no lake -> empty board, not an error
            return pd.DataFrame(columns=_COLS)
    rows = []

    kick = build_kicker(con)
    if not kick.empty:
        scored = score_kicker(kick, rules)
        pred = _trailing_pred(scored, "player_display_name", season, week, window)
        if not pred.empty:
            team = (scored.sort_values(["season", "week"])
                    .groupby("player_display_name")["recent_team"].last())
            k = pred.assign(position="K",
                            recent_team=pred["player_display_name"].map(team).fillna(""))
            rows.append(k[_COLS])

    dst = build_dst(con)
    if not dst.empty:
        scored = score_dst(dst, rules)
        pred = _trailing_pred(scored, "team", season, week, window)
        if not pred.empty:
            d = pred.rename(columns={"team": "recent_team"})
            d["player_display_name"] = d["recent_team"] + " DST"
            d["position"] = "DEF"
            rows.append(d[_COLS])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=_COLS)
