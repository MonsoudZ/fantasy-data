"""Local web UI for the lineup optimizer.

    python -m ffdata.web      # -> http://127.0.0.1:8000

A thin FastAPI wrapper: the browser posts a roster + scoring + week, the server
projects that week and runs the optimizer, and returns the recommended lineup.
The heavy work (fitting the projection model + residual sampler) is cached per
(scoring, projector) so only the first request for a config is slow (~2 min);
after that, projections and optimization are seconds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .ingest import current_nfl_season
from .matchup import MatchupSimulator
from .optimize import LineupOptimizer, _assemble, _match
from .scoring import PPR, HALF_PPR, STANDARD

_RULES = {"ppr": PPR, "half": HALF_PPR, "standard": STANDARD}
_SIMS: dict = {}
_BOARDS: dict = {}
_STATIC = Path(__file__).parent / "static"


def _sim(scoring: str, projector: str) -> MatchupSimulator:
    key = (scoring, projector)
    if key not in _SIMS:
        _SIMS[key] = MatchupSimulator.fit(projector=projector, rules=_RULES[scoring])
    return _SIMS[key]


def _board(scoring: str, projector: str, season: int, week: int):
    key = (scoring, projector, season, week)
    if key not in _BOARDS:
        _BOARDS[key] = _sim(scoring, projector).project(season, week).reset_index(drop=True)
    return _sim(scoring, projector), _BOARDS[key]


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
    season: int = current_nfl_season()
    week: int
    scoring: str = "ppr"
    projector: str = "gbm"
    mode: str = "h2h"
    roster: str = ""
    opponent: str = ""
    full_slate: bool = False
    ceiling: float = 0.90
    stack_size: int = 2
    bringback: int = 1


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/api/config")
def config():
    return {"season": current_nfl_season()}


@app.post("/api/players")
def players(req: OptRequest):
    """The full projection board for a week -- every projectable player."""
    try:
        _, board = _board(req.scoring, req.projector, req.season, req.week)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    if board.empty:
        return {"ok": False, "error": f"No data for {req.season} week {req.week}."}
    rows = [{"name": r["player_display_name"], "position": r["position"],
             "team": str(r.get("recent_team", "")), "pred": round(float(r["pred"]), 1)}
            for _, r in board.iterrows()]
    return {"ok": True, "count": len(rows), "players": rows}


@app.post("/api/optimize")
def optimize(req: OptRequest):
    if req.scoring not in _RULES or req.projector not in ("gbm", "neural"):
        return {"ok": False, "error": "bad scoring or projector"}
    try:
        sim, board = _board(req.scoring, req.projector, req.season, req.week)
    except Exception as exc:  # noqa: BLE001 - surface to the UI
        return {"ok": False, "error": f"could not build projections: {exc}"}
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

    opt = LineupOptimizer(sim)
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
        res = opt.optimize(pool, _assemble(opp))
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


def main() -> None:
    import uvicorn
    print("\n  ff-data lineup optimizer  ->  http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
