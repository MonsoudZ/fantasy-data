"""Local web UI for the lineup optimizer.

    python -m ffdata.web      # -> http://127.0.0.1:8000

A thin FastAPI wrapper: the browser posts a roster + scoring + week, the server
projects that week and runs the optimizer, and returns the recommended lineup.
The heavy work (fitting the projection model + residual sampler) is cached per
(scoring, projector) so only the first request for a config is slow (~2 min);
after that, projections and optimization are seconds.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import advice
from .draft import (DEFAULT_LEAGUE, best_available, draft_board, keeper_value,
                    rookie_context, trade_value)
from .dynasty import dynasty_board
from .features import build_features
from .gamelines import game_forecasts
from .ingest import FIRST_SEASON, current_nfl_season, upcoming_nfl_season
from .kdst import project_kdst
from .matchup import MatchupSimulator
from .optimize import (
    LineupOptimizer, _assemble, _match, _norm, free_agent_advice, slots_from_lineup,
)
from .props import price_props
from .scoring import PPR, HALF_PPR, STANDARD, preset_name, rules_from, rules_to_dict
from .sleeper import import_league, list_user_leagues
from .store import (
    League, Team, delete_league, delete_team, list_leagues, list_teams,
    save_league, save_team,
)

_log = logging.getLogger("ffdata.web")

_RULES = {"ppr": PPR, "half": HALF_PPR, "standard": STANDARD}
_SIMS: dict = {}
_BOARDS: dict = {}
_DRAFT: dict = {}
_DYN: dict = {}
_GAMES: dict = {}
_FEATS: dict = {}
_STATIC = Path(__file__).parent / "static"

# Each fitted simulator/board/feature frame is large. Cap the per-config caches
# so a stream of distinct (scoring, projector, season, week) requests can't grow
# memory without bound; evict the oldest entry (dicts preserve insertion order).
_MAX_CACHE = 16


def _cache_put(cache: dict, key, value):
    if key not in cache and len(cache) >= _MAX_CACHE:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def _scoring_key(scoring: str, rules: dict | None):
    """A hashable, stable cache key for a scoring config (preset or custom)."""
    return ("custom", tuple(sorted(rules.items()))) if rules else scoring


def _sim(scoring_key, projector: str, rules_obj) -> MatchupSimulator:
    key = (scoring_key, projector)
    if key not in _SIMS:
        _cache_put(_SIMS, key, MatchupSimulator.fit(projector=projector, rules=rules_obj))
    return _SIMS[key]


def _board(scoring: str, projector: str, season: int, week: int, rules: dict | None = None):
    rules_obj = rules_from(scoring, rules)
    sk = _scoring_key(scoring, rules)
    sim = _sim(sk, projector, rules_obj)
    key = (sk, projector, season, week)
    if key not in _BOARDS:
        board = sim.project(season, week).reset_index(drop=True)
        # Append kicker + team-defense projections so standard leagues (which
        # start a K and a DEF) can fill those slots. Empty (a no-op) when the K/
        # DST data isn't ingested -- the skill board still works on its own.
        kd = project_kdst(season, week, rules=rules_obj)
        if not kd.empty:
            board = pd.concat([board, kd], ignore_index=True)
        _cache_put(_BOARDS, key, board)
    return sim, _BOARDS[key]


def _names(text: str) -> list[str]:
    return [ln.strip().split(",")[0].strip() for ln in text.splitlines() if ln.strip()]


def _rows(lineup, board: pd.DataFrame) -> list[dict]:
    b = board.set_index("player_display_name")
    out = []
    for slot, name in lineup:
        r = b.loc[name]
        out.append({"slot": slot, "name": name, "position": r["position"],
                    "team": str(r.get("recent_team", "")), "pred": round(float(r["pred"]), 1)})
    return out


app = FastAPI(title="ff-data lineup optimizer")


class OptRequest(BaseModel):
    season: int = Field(default_factory=current_nfl_season, ge=1999, le=2100)
    week: int = Field(ge=1, le=22)
    scoring: str = "ppr"
    projector: str = "gbm"
    mode: str = "h2h"
    roster: str = ""
    opponent: str = ""
    full_slate: bool = False
    ceiling: float = Field(0.90, ge=0.5, lt=1.0)
    stack_size: int = Field(2, ge=1, le=5)
    bringback: int = Field(1, ge=0, le=3)
    rules: dict | None = None    # full custom scoring (from an imported league)
    lineup: dict | None = None   # {starters, flex, superflex} for superflex slots


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/api/config")
def config():
    return {"season": current_nfl_season(), "draft_season": upcoming_nfl_season(),
            "advice": advice.available()}


@app.post("/api/players")
def players(req: OptRequest):
    """The full projection board for a week -- every projectable player."""
    try:
        _, board = _board(req.scoring, req.projector, req.season, req.week, req.rules)
    except Exception:  # noqa: BLE001 - log detail server-side, keep the UI generic
        _log.exception("projection board failed (%s)", req)
        return {"ok": False, "error": "could not build the projection board (see server logs)"}
    if board.empty:
        return {"ok": False, "error": f"No data for {req.season} week {req.week}."}
    rows = [{"name": r["player_display_name"], "position": r["position"],
             "team": str(r.get("recent_team", "")), "pred": round(float(r["pred"]), 1)}
            for _, r in board.iterrows()]
    return {"ok": True, "count": len(rows), "players": rows}


@app.post("/api/optimize")
def optimize(req: OptRequest):
    if req.projector not in ("gbm", "neural") or (req.scoring not in _RULES and not req.rules):
        return {"ok": False, "error": "bad scoring or projector"}
    try:
        sim, board = _board(req.scoring, req.projector, req.season, req.week, req.rules)
    except Exception:  # noqa: BLE001 - log detail server-side, keep the UI generic
        _log.exception("projection build failed (%s)", req)
        return {"ok": False, "error": "could not build projections (see server logs)"}
    if board.empty:
        return {"ok": False, "error": f"No projections for {req.season} week {req.week}. "
                f"Ingest it first: python -m ffdata.cli --seasons {req.season}"}

    if req.full_slate:
        # DFS: optimize over the whole slate. Cap by projection to bound compute.
        pool, missing = board.nlargest(160, "pred").reset_index(drop=True), []
    else:
        pool, missing = _match(_names(req.roster), board)
        if pool.empty:
            return {"ok": False, "error": "None of your roster names matched the projection board.",
                    "missing": missing}

    slots = slots_from_lineup(req.lineup)
    opt = LineupOptimizer(sim, slots=slots)
    if req.mode == "tournament":
        res = opt.optimize_tournament(pool, quantile=req.ceiling)["optimal"]
        headline = {"label": f"{int(req.ceiling*100)}th-pct ceiling", "value": f"{res['ceiling']}",
                    "sub": f"proj {res['proj']} · median {res['median']}"}
        lineup, note = res["lineup"], "Optimized for tournament ceiling (upside)."
    elif req.mode == "stack":
        res = opt.optimize_game_stack(pool, quantile=req.ceiling,
                                      stack_size=req.stack_size, bringback=req.bringback)["optimal"]
        st = res.get("stack")
        sub = (f"stack: {st['team']} {st['qb']} + {len(st['members'])-1} more" if st else f"proj {res['proj']}")
        headline = {"label": f"{int(req.ceiling*100)}th-pct ceiling", "value": f"{res['ceiling']}", "sub": sub}
        lineup, note = res["lineup"], "Best lineup built around a QB game stack."
    elif req.opponent.strip():
        opp, opp_missing = _match(_names(req.opponent), board)
        missing += [f"(opp) {m}" for m in opp_missing]
        res = opt.optimize(pool, _assemble(opp, slots))
        headline = {"label": "Win probability", "value": f"{res['optimal_win_prob']*100:.1f}%",
                    "sub": f"you {res['optimal_proj']} vs opp {res['opp_proj']}"}
        lineup, note = res["optimal_lineup"], "Optimized for win probability, not just points."
    else:
        lineup = [(s, n) for s, n, _, _ in opt._greedy_points(pool)]
        proj = round(sum(r["pred"] for r in _rows(lineup, board)), 1)
        headline = {"label": "Projected points", "value": f"{proj}", "sub": "highest-projected starters"}
        note = "Add an opponent for win-probability optimization, or pick a tournament mode."

    rows = _rows(lineup, board)
    return {"ok": True, "mode": req.mode, "headline": headline, "lineup": rows,
            "total": round(sum(r["pred"] for r in rows), 1), "note": note,
            "missing": missing, "pool_size": len(pool)}


class FreeAgentRequest(BaseModel):
    season: int = Field(default_factory=current_nfl_season, ge=1999, le=2100)
    week: int = Field(ge=1, le=22)
    scoring: str = "ppr"
    projector: str = "gbm"
    roster: str = ""             # your players, one name per line
    exclude: str = ""            # players rostered by others (optional), one per line
    rules: dict | None = None
    lineup: dict | None = None   # {starters, flex, superflex} for superflex slots
    n: int = Field(15, ge=1, le=50)


@app.post("/api/freeagents")
def api_freeagents(req: FreeAgentRequest):
    """Rank available players by how much they'd upgrade your starting lineup."""
    if req.projector not in ("gbm", "neural") or (req.scoring not in _RULES and not req.rules):
        return {"ok": False, "error": "bad scoring or projector"}
    roster = _names(req.roster)
    if not roster:
        return {"ok": False, "error": "Add your roster first (one player per line)."}
    try:
        _, board = _board(req.scoring, req.projector, req.season, req.week, req.rules)
    except Exception:  # noqa: BLE001 - log detail server-side, keep the UI generic
        _log.exception("free-agent board failed (%s)", req)
        return {"ok": False, "error": "could not build projections (see server logs)"}
    if board.empty:
        return {"ok": False, "error": f"No projections for {req.season} week {req.week}."}
    res = free_agent_advice(board, roster, slots=slots_from_lineup(req.lineup),
                            exclude=_names(req.exclude), top=req.n)
    matched, missing = _match(roster, board)
    return {"ok": True, **res, "missing": missing, "roster_matched": len(matched)}


class BoardRequest(BaseModel):
    """Shared draft-board config (draft board, keepers, and trades all use it)."""
    # Drafting targets the UPCOMING season -- current_nfl_season() is the most
    # recently *played* one, which in the offseason is already over.
    season: int = Field(default_factory=upcoming_nfl_season, ge=1999, le=2100)
    teams: int = Field(12, ge=2, le=32)
    scoring: str = "ppr"
    rules: dict | None = None    # full custom scoring (from an imported league)
    lineup: dict | None = None   # {starters, flex, superflex} for VOR (imported)


class DraftRequest(BoardRequest):
    drafted: list[str] = []
    position: str | None = None
    n: int = Field(50, ge=1, le=500)


class KeeperRequest(BoardRequest):
    keepers: list = []           # [[player, cost], ...]
    cost_type: str = "auction"   # "auction" ($) or "round" (draft round)


class TradeRequest(BoardRequest):
    side_a: list[str] = []
    side_b: list[str] = []


def _league_cfg(teams: int, lineup: dict | None) -> dict:
    """A draft-league config: default lineup unless an imported one is supplied."""
    cfg = {**DEFAULT_LEAGUE, "teams": teams, "superflex": 0}
    if lineup:
        cfg["starters"] = {**DEFAULT_LEAGUE["starters"], **(lineup.get("starters") or {})}
        cfg["flex"] = int(lineup.get("flex", DEFAULT_LEAGUE["flex"]))
        cfg["superflex"] = int(lineup.get("superflex", 0))
    return cfg


def _lineup_key(lineup: dict | None):
    if not lineup:
        return None
    return (tuple(sorted((lineup.get("starters") or {}).items())),
            lineup.get("flex"), lineup.get("superflex"))


def _get_board(req: BoardRequest):
    """Build (or reuse the cached) draft board for a BoardRequest's config.

    Raises ValueError('bad scoring') on invalid scoring; other failures (e.g. no
    data lake) propagate for the caller to log and surface generically.
    """
    if req.scoring not in _RULES and not req.rules:
        raise ValueError("bad scoring")
    key = (req.season, req.teams, _scoring_key(req.scoring, req.rules), _lineup_key(req.lineup))
    if key not in _DRAFT:
        _cache_put(_DRAFT, key, draft_board(
            req.season, _league_cfg(req.teams, req.lineup),
            rules=rules_from(req.scoring, req.rules)))
    return _DRAFT[key]


def _board_or_error(req: BoardRequest):
    """(board, None) on success, or (None, error_response) to return to the UI."""
    try:
        board = _get_board(req)
    except ValueError as exc:
        return None, {"ok": False, "error": str(exc)}
    except Exception:  # noqa: BLE001 - log detail server-side, keep the UI generic
        _log.exception("draft_board failed (season=%s teams=%s)", req.season, req.teams)
        return None, {"ok": False, "error": "could not build the draft board (see server logs)"}
    if board.empty:
        return None, {"ok": False, "error": f"No draftable data for {req.season}. Ingest it first."}
    return board, None


@app.post("/api/draft")
def api_draft(req: DraftRequest):
    board, err = _board_or_error(req)
    if err:
        return err
    avail = best_available(board, req.drafted, req.position or None, req.n)
    ctx = _rookie_ctx(req.season)
    players = []
    for _, r in avail.iterrows():
        row = {"player": r["player"], "position": r["position"], "proj": round(float(r["proj"]), 1),
               "vor": round(float(r["vor"]), 1), "auction": int(r["auction"])}
        # Rookies carry situation context -- the projection only knows draft
        # capital, so the room they land in is the drafter's call to weigh.
        if r["player"] in ctx:
            row["rookie"] = ctx[r["player"]]
        players.append(row)
    return {"ok": True, "count": len(players), "total": len(board), "players": players}


@lru_cache(maxsize=8)
def _rookie_ctx(season: int) -> dict:
    """player -> opportunity context (vacated, who blocks him, depth, pass rate)."""
    try:
        c = rookie_context(season)
    except Exception:  # noqa: BLE001 - context is a bonus, never fatal
        return {}
    if c is None or c.empty:
        return {}
    out = {}
    for _, r in c.iterrows():
        out[r["player"]] = {
            "pick": int(r["pick"]),
            "team": r["team"],
            "vacated": float(r["vacated_fp"]),
            "blocked_by": (None if pd.isna(r["blocked_by"]) else str(r["blocked_by"])),
            "blocked_by_fp": float(r["blocked_by_fp"]),
            "depth": (None if pd.isna(r["depth_rank"]) else int(r["depth_rank"])),
            "pass_rate": (None if pd.isna(r["pass_rate"]) else float(r["pass_rate"])),
        }
    return out


def _keeper_pairs(keepers: list) -> list[tuple[str, float]]:
    """Clean [[player, cost], ...] into typed (name, cost) tuples, dropping junk."""
    pairs = []
    for k in keepers:
        if isinstance(k, (list, tuple)) and len(k) >= 2:
            try:
                pairs.append((str(k[0]), float(k[1])))
            except (ValueError, TypeError):
                pass
    return pairs


@app.post("/api/keepers")
def api_keepers(req: KeeperRequest):
    board, err = _board_or_error(req)
    if err:
        return err
    pairs = _keeper_pairs(req.keepers)
    if not pairs:
        return {"ok": False, "error": "No valid keepers. Format: player, cost"}
    cost_type = req.cost_type if req.cost_type in ("auction", "round") else "auction"
    df = keeper_value(board, pairs, teams=req.teams, cost_type=cost_type)
    return {"ok": True, "cost_type": cost_type, "keepers": df.to_dict("records")}


@app.post("/api/trade")
def api_trade(req: TradeRequest):
    board, err = _board_or_error(req)
    if err:
        return err
    if not req.side_a and not req.side_b:
        return {"ok": False, "error": "Add players to at least one side."}
    return {"ok": True, **trade_value(board, req.side_a, req.side_b)}


class CompareRequest(BoardRequest):
    players: list[str] = []


def _compare_rows(board: pd.DataFrame, names: list[str]):
    """(rows, missing) for the named players, each with overall/positional rank.

    draft_board is sorted by VOR desc, so row order == overall rank; positional
    rank is the running count within each position in that same order.
    """
    board = board.reset_index(drop=True)
    index, pos_seen = {}, {}
    for i, r in board.iterrows():
        pos_seen[r["position"]] = pos_seen.get(r["position"], 0) + 1
        index[_norm(r["player"])] = (int(i) + 1, pos_seen[r["position"]], r)

    out, missing = [], []
    for name in names:
        hit = index.get(_norm(name))
        if hit is None:
            missing.append(name)
            continue
        overall, pos_rank, r = hit
        out.append({"player": r["player"], "position": r["position"],
                    "proj": round(float(r["proj"]), 1), "vor": round(float(r["vor"]), 1),
                    "auction": int(r["auction"]),
                    "overall_rank": overall, "position_rank": pos_rank})
    return out, missing


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    board, err = _board_or_error(req)
    if err:
        return err
    names = [n for n in req.players if str(n).strip()][:3]
    if len(names) < 2:
        return {"ok": False, "error": "Pick at least 2 players to compare."}
    out, missing = _compare_rows(board, names)
    if not out:
        return {"ok": False, "error": "None of those players are on the board.", "missing": missing}
    best = max(out, key=lambda p: p["vor"])["player"]     # highest VOR = best value
    return {"ok": True, "players": out, "best_value": best, "missing": missing}


class AdviceRequest(BoardRequest):
    """A grounded-explanation request over one decision (compare/keeper/trade).

    Carries the board config (from BoardRequest) plus every kind's inputs; only
    the fields for the chosen `kind` are read, so the UI can post the tool's
    current state verbatim.
    """
    kind: str                        # "compare" | "keeper" | "trade"
    players: list[str] = []          # compare
    keepers: list = []               # keeper
    cost_type: str = "auction"       # keeper
    side_a: list[str] = []           # trade
    side_b: list[str] = []           # trade


def _scoring_facts(req: BoardRequest) -> dict:
    """The league-scoring context every advice call is grounded in."""
    rules = rules_from(req.scoring, req.rules)
    cfg = _league_cfg(req.teams, req.lineup)
    return {"scoring": preset_name(rules), "rules": rules_to_dict(rules),
            "teams": req.teams, "superflex": bool(cfg.get("superflex"))}


def _advice_facts(req: AdviceRequest, board: pd.DataFrame):
    """Build (facts, error) for a request kind by reusing the engine outputs."""
    if req.kind == "compare":
        names = [n for n in req.players if str(n).strip()][:3]
        if len(names) < 2:
            return None, "Pick at least 2 players to compare."
        rows, _ = _compare_rows(board, names)
        if len(rows) < 2:
            return None, "Need at least 2 of those players on the board."
        return {"decision": "compare", "players": rows}, None
    if req.kind == "keeper":
        pairs = _keeper_pairs(req.keepers)
        if not pairs:
            return None, "No valid keepers. Format: player, cost"
        cost_type = req.cost_type if req.cost_type in ("auction", "round") else "auction"
        df = keeper_value(board, pairs, teams=req.teams, cost_type=cost_type)
        if df.empty:
            return None, "None of those keepers are on the board."
        return {"decision": "keeper", "cost_type": cost_type,
                "keepers": df.to_dict("records")}, None
    if req.kind == "trade":
        if not req.side_a and not req.side_b:
            return None, "Add players to at least one side."
        tv = trade_value(board, req.side_a, req.side_b)
        if not tv["side_a"]["players"] and not tv["side_b"]["players"]:
            return None, "None of those players are on the board."
        return {"decision": "trade", **tv}, None
    return None, f"unknown advice kind: {req.kind}"


@app.post("/api/advice")
def api_advice(req: AdviceRequest):
    """A grounded, plain-English read of a compare/keeper/trade decision.

    Reuses the same board + engine functions the tools do, hands the resulting
    numbers (plus scoring context) to the advice layer, and returns its text.
    """
    if not advice.available():
        return {"ok": False, "error": "Advice is off. Install '.[advice]' and set "
                "ANTHROPIC_API_KEY to enable it."}
    board, err = _board_or_error(req)
    if err:
        return err
    facts, ferr = _advice_facts(req, board)
    if ferr:
        return {"ok": False, "error": ferr}
    facts = {**facts, **_scoring_facts(req)}
    try:
        text = advice.explain(req.kind, facts)
    except Exception:  # noqa: BLE001 - log detail server-side, keep the UI generic
        _log.exception("advice.explain failed (kind=%s)", req.kind)
        return {"ok": False, "error": "could not generate advice (see server logs)"}
    return {"ok": True, "kind": req.kind, "advice": text}


class DynastyRequest(BoardRequest):
    years: int = Field(4, ge=1, le=10)
    discount: float = Field(0.85, ge=0.5, le=1.0)
    drafted: list[str] = []
    position: str | None = None
    n: int = Field(50, ge=1, le=500)


@app.post("/api/dynasty")
def api_dynasty(req: DynastyRequest):
    if req.scoring not in _RULES and not req.rules:
        return {"ok": False, "error": "bad scoring"}
    key = (req.season, req.teams, _scoring_key(req.scoring, req.rules),
           _lineup_key(req.lineup), req.years, req.discount)
    try:
        if key not in _DYN:
            _cache_put(_DYN, key, dynasty_board(
                req.season, years=req.years, discount=req.discount,
                rules=rules_from(req.scoring, req.rules),
                league=_league_cfg(req.teams, req.lineup)))
    except Exception:  # noqa: BLE001 - log detail server-side, keep the UI generic
        _log.exception("dynasty_board failed (season=%s)", req.season)
        return {"ok": False, "error": "could not build the dynasty board (see server logs)"}
    board = _DYN[key]
    if board.empty:
        return {"ok": False, "error": f"No dynasty data for {req.season}. Ingest it first."}
    taken = {_norm(x) for x in (req.drafted or [])}
    out = board[~board["player"].map(lambda s: _norm(s) in taken)]
    if req.position:
        out = out[out["position"] == req.position]
    players = [{"player": r["player"], "position": r["position"], "age": int(r["age"]),
                "proj": round(float(r["proj"]), 1), "vor": round(float(r["vor"]), 1),
                "dynasty_value": round(float(r["dynasty_value"]), 1)}
               for _, r in out.head(req.n).iterrows()]
    return {"ok": True, "count": len(players), "total": len(board), "players": players}


class GamesRequest(BaseModel):
    season: int = Field(default_factory=current_nfl_season, ge=1999, le=2100)
    week: int = Field(ge=1, le=22)


@app.post("/api/games")
def api_games(req: GamesRequest):
    """Game-line forecasts vs the market for a week (totals/spreads/moneylines)."""
    key = (req.season, req.week)
    try:
        if key not in _GAMES:
            _cache_put(_GAMES, key, game_forecasts(req.season, req.week))
    except Exception:  # noqa: BLE001 - log detail server-side, keep the UI generic
        _log.exception("game_forecasts failed (%s)", key)
        return {"ok": False, "error": "could not build game forecasts (see server logs)"}
    board = _GAMES[key]
    if board.empty:
        return {"ok": False, "error": f"No game lines for {req.season} week {req.week}. "
                "Ingest schedules first."}
    return {"ok": True, "count": len(board), "games": board.to_dict("records")}


class PropsRequest(BaseModel):
    season: int = Field(default_factory=current_nfl_season, ge=1999, le=2100)
    week: int = Field(ge=1, le=22)
    lines: str = ""


def _parse_props(text: str) -> pd.DataFrame:
    rows = []
    for i, line in enumerate(text.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        if i == 0 and parts[1].lower() in ("market", "prop"):
            continue
        try:
            rows.append({"player": parts[0], "market": parts[1], "line": float(parts[2]),
                         "over_odds": float(parts[3]), "under_odds": float(parts[4])})
        except ValueError:
            continue
    return pd.DataFrame(rows)


@app.post("/api/props")
def api_props(req: PropsRequest):
    prop_df = _parse_props(req.lines)
    if prop_df.empty:
        return {"ok": False, "error": "No valid prop lines. Format: player,market,line,over_odds,under_odds"}
    try:
        if req.season not in _FEATS:
            _cache_put(_FEATS, req.season,
                       build_features(seasons=list(range(FIRST_SEASON, req.season + 1))))
        edges = price_props(prop_df, req.season, req.week, feats=_FEATS[req.season], threshold=-999)
    except Exception:  # noqa: BLE001 - log detail server-side, keep the UI generic
        _log.exception("price_props failed (season=%s week=%s)", req.season, req.week)
        return {"ok": False, "error": "could not price props (see server logs)"}
    return {"ok": True, "priced": len(edges),
            "edges": edges.to_dict("records"),
            "markets": sorted({m for m in prop_df["market"].unique()})}


# --------------------------------------------------------------------------- #
# Saved leagues (persisted config + draft state; see ffdata/store.py)
# --------------------------------------------------------------------------- #

class LeagueModel(BaseModel):
    name: str
    season: int = Field(ge=1999, le=2100)
    scoring: str = "ppr"
    teams: int = Field(12, ge=2, le=32)
    drafted: list[str] = []
    keepers: list = []
    rules: dict | None = None
    lineup: dict | None = None


class LeagueName(BaseModel):
    name: str


@app.get("/api/leagues")
def api_leagues():
    return {"ok": True, "leagues": [asdict(lg) for lg in list_leagues()]}


@app.post("/api/leagues")
def api_league_save(req: LeagueModel):
    try:
        lg = save_league(League(**req.model_dump()))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "league": asdict(lg)}


@app.post("/api/leagues/delete")
def api_league_delete(req: LeagueName):
    return {"ok": True, "deleted": delete_league(req.name)}


class TeamModel(BaseModel):
    name: str
    season: int = Field(ge=1999, le=2100)
    scoring: str = "ppr"
    projector: str = "gbm"
    roster: dict = Field(default_factory=lambda: {"QB": [], "RB": [], "WR": [], "TE": [], "K": [], "DEF": []})
    rules: dict | None = None


@app.get("/api/teams")
def api_teams():
    return {"ok": True, "teams": [asdict(t) for t in list_teams()]}


@app.post("/api/teams")
def api_team_save(req: TeamModel):
    try:
        t = save_team(Team(**req.model_dump()))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "team": asdict(t)}


@app.post("/api/teams/delete")
def api_team_delete(req: LeagueName):
    return {"ok": True, "deleted": delete_team(req.name)}


# --------------------------------------------------------------------------- #
# Import from Sleeper (public read-only API; see ffdata/sleeper.py)
# --------------------------------------------------------------------------- #

class SleeperUser(BaseModel):
    username: str
    season: int = Field(default_factory=current_nfl_season, ge=1999, le=2100)


class SleeperImport(BaseModel):
    league_id: str
    username: str
    season: int = Field(default_factory=current_nfl_season, ge=1999, le=2100)


@app.post("/api/import/sleeper/leagues")
def api_sleeper_leagues(req: SleeperUser):
    """List a Sleeper user's leagues for a season, so the UI can offer a picker."""
    try:
        leagues = list_user_leagues(req.username, req.season)
    except Exception:  # noqa: BLE001 - log detail server-side, keep the UI generic
        _log.exception("sleeper league list failed (%s)", req.username)
        return {"ok": False, "error": "could not reach Sleeper (check the username and your network)"}
    if not leagues:
        return {"ok": False, "error": f"No Sleeper leagues found for '{req.username}' in {req.season}."}
    return {"ok": True, "leagues": leagues}


@app.post("/api/import/sleeper/league")
def api_sleeper_import(req: SleeperImport):
    """Import one Sleeper league -> save it as a League (settings/scoring/drafted)
    and a Team (your roster). Both then appear in the saved dropdowns."""
    try:
        league, team = import_league(req.league_id, req.username, req.season)
        save_league(league)
        save_team(team)
    except Exception:  # noqa: BLE001 - log detail server-side, keep the UI generic
        _log.exception("sleeper import failed (%s)", req.league_id)
        return {"ok": False, "error": "could not import that league from Sleeper"}
    return {"ok": True, "league": league.name, "team": team.name,
            "scoring": league.scoring, "teams": league.teams,
            "drafted": len(league.drafted),
            "roster_size": sum(len(v) for v in team.roster.values())}


def main() -> None:
    import uvicorn
    print("\n  ff-data lineup optimizer  ->  http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
