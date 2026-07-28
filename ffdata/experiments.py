"""CLI runners for reproducible weekly, draft, and season experiments.

Examples:
    python -m ffdata.experiments weekly --holdout regression
    python -m ffdata.experiments draft --holdout 2024
    python -m ffdata.experiments season --holdout regression --opponent naive
    python -m ffdata.experiments season-sweep --holdout regression
    python -m ffdata.experiments summary
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .experiment_registry import (
    HoldoutError,
    bootstrap_cluster_mean_ci,
    bootstrap_mean_ci,
    data_metadata,
    execute,
    git_metadata,
    load_policy,
    render_summary,
    resolve_experiment_root,
    resolve_holdout,
    save_record,
)
from .scoring import HALF_PPR, PPR, STANDARD

_RULES = {"ppr": PPR, "half": HALF_PPR, "standard": STANDARD}


def _ensure_played(season: int) -> None:
    from .db import connect
    from .ingest import season_not_started
    con = connect()
    if season_not_started(season, con=con):
        raise HoldoutError(f"{season} has no played weekly data yet; the holdout is still sealed")


def weekly_metrics(season: int, train_from: int = 2019,
                   positions: tuple[str, ...] = ("QB", "RB", "WR", "TE")) -> dict:
    from .projections import backtest
    table = backtest(train_from=train_from, test_seasons=[season], positions=positions)
    return {"models": table.to_dict(orient="index")}


def weekly_config(train_from: int, positions: tuple[str, ...]) -> dict:
    from .projections import GBMProjector, TrailingAverageProjector
    baseline, model = TrailingAverageProjector(), GBMProjector()
    return {
        "train_from": train_from,
        "positions": positions,
        "models": {
            baseline.name: {"w3": baseline.w3, "w5": baseline.w5},
            model.name: model.params,
        },
    }


def draft_metrics(season: int, scoring: str = "ppr", realistic: bool = True) -> dict:
    """Rank the shipped realistic board, with a bare-model comparison available."""
    from scipy.stats import spearmanr

    from .db import connect
    from .draft import _feature_frame, _season_agg, draft_board, project_season

    con, rules = connect(), _RULES[scoring]
    if realistic:
        proj = draft_board(
            season, rules=rules, con=con, include_rookies=True,
            career=True, competition=True, reconcile=True,
        )[["player_id", "proj"]]
    else:
        proj = project_season(season, rules=rules, con=con)[["player_id", "proj"]]
    actual = _season_agg(con, rules)
    actual = actual[actual["season"] == season][["player_id", "fp"]]
    all_players = proj.merge(actual, on="player_id", how="inner")
    naive = _feature_frame(con, rules)
    naive = naive[naive["tseason"] == season][["player_id", "p_fp"]]
    shared = all_players.merge(naive, on="player_id", how="inner").dropna(subset=["p_fp"])

    def rank(a, b) -> float | None:
        if len(a) < 2 or a.nunique() < 2 or b.nunique() < 2:
            return None
        return round(float(spearmanr(a, b).correlation), 4)

    return {
        "season": season,
        "overall_n": len(all_players),
        "overall_model_spearman": rank(all_players["proj"], all_players["fp"]),
        "n": len(shared),
        "model_spearman": rank(shared["proj"], shared["fp"]),
        "lastyear_spearman": rank(shared["p_fp"], shared["fp"]),
        "model_mae": round(float((all_players["proj"] - all_players["fp"]).abs().mean()), 2),
    }


def season_metrics(
    season: int,
    *,
    scoring: str = "standard",
    teams: int = 12,
    projector: str = "gbm",
    waivers: bool = True,
    opponent: str = "naive",
    noise: float = 24.0,
    sharp_fraction: float = 0.5,
    seed: int = 0,
    bootstrap_resamples: int = 2000,
) -> dict:
    from .season_sim import run_all_slots
    result = run_all_slots(
        season, rules=_RULES[scoring], n_teams=teams, projector=projector,
        waivers=waivers, opponent=opponent, noise=noise,
        sharp_fraction=sharp_fraction, log=lambda *a: None,
    )
    runs = result["runs"]
    places = [float(r["regular_season_place"]) for r in runs]
    playoffs = [float(p <= 6) for p in places]
    titles = [float(r["we_won"]) for r in runs]
    return {
        "n_slots": len(runs),
        "places": [int(p) for p in places],
        "titles": int(sum(titles)),
        "title_rate": round(sum(titles) / len(titles), 4),
        "mean_finish": round(sum(places) / len(places), 4),
        "playoff_rate": round(sum(playoffs) / len(playoffs), 4),
        "confidence_intervals": {
            "mean_finish": bootstrap_mean_ci(
                places, resamples=bootstrap_resamples, seed=seed,
            ),
            "playoff_rate": bootstrap_mean_ci(
                playoffs, resamples=bootstrap_resamples, seed=seed + 1,
            ),
            "title_rate": bootstrap_mean_ci(
                titles, resamples=bootstrap_resamples, seed=seed + 2,
            ),
        },
        "bootstrap": {
            "method": "percentile",
            "sampling_unit": "draft_slot_within_one_season",
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "seed": seed,
        },
    }


def _parse_fractions(spec: str) -> list[float]:
    fractions = sorted({float(value.strip()) for value in spec.split(",") if value.strip()})
    if not fractions:
        raise HoldoutError("at least one sharp fraction is required")
    if fractions[0] < 0 or fractions[-1] > 1:
        raise HoldoutError("sharp fractions must be between 0 and 1")
    if 0.0 not in fractions:
        raise HoldoutError("season sweeps require a 0.0 sharp baseline")
    return fractions


def season_sweep_metrics(
    seasons: list[int],
    sharp_fractions: list[float],
    *,
    scoring: str = "standard",
    teams: int = 12,
    projector: str = "gbm",
    waivers: bool = True,
    noise: float = 24.0,
    seed: int = 0,
    bootstrap_resamples: int = 2000,
) -> dict:
    """Run paired draft-slot replays across seasons and mixed field strengths."""
    from .season_sim import prepare, run_all_slots

    rules = _RULES[scoring]
    runs_by_fraction: dict[float, list[list[dict]]] = {
        fraction: [] for fraction in sharp_fractions
    }
    for season in seasons:
        ctx = prepare(
            season, rules=rules, projector=projector, n_teams=teams,
            log=lambda *a: None,
        )
        for fraction in sharp_fractions:
            result = run_all_slots(
                season, rules=rules, n_teams=teams, projector=projector,
                waivers=waivers, opponent="mixed", sharp_fraction=fraction,
                noise=noise, ctx=ctx, log=lambda *a: None,
            )
            runs_by_fraction[fraction].append(result["runs"])

    baseline = runs_by_fraction[0.0]
    fields = []
    for index, fraction in enumerate(sharp_fractions):
        season_runs = runs_by_fraction[fraction]
        place_clusters = [
            [float(run["regular_season_place"]) for run in runs]
            for runs in season_runs
        ]
        playoff_clusters = [[float(place <= 6) for place in places]
                            for places in place_clusters]
        title_clusters = [[float(run["we_won"]) for run in runs]
                          for runs in season_runs]
        places = [value for cluster in place_clusters for value in cluster]
        playoffs = [value for cluster in playoff_clusters for value in cluster]
        titles = [value for cluster in title_clusters for value in cluster]
        row = {
            "sharp_fraction": fraction,
            "sharp_opponents": int(((teams - 1) * fraction) + 0.5),
            "n_slots": len(places),
            "mean_finish": round(sum(places) / len(places), 4),
            "playoff_rate": round(sum(playoffs) / len(playoffs), 4),
            "titles": int(sum(titles)),
            "title_rate": round(sum(titles) / len(titles), 4),
            "by_season": [
                {
                    "season": season,
                    "places": [int(place) for place in place_clusters[i]],
                    "mean_finish": round(sum(place_clusters[i]) / len(place_clusters[i]), 4),
                    "playoff_rate": round(
                        sum(playoff_clusters[i]) / len(playoff_clusters[i]), 4,
                    ),
                    "titles": int(sum(title_clusters[i])),
                    "title_rate": round(sum(title_clusters[i]) / len(title_clusters[i]), 4),
                }
                for i, season in enumerate(seasons)
            ],
            "confidence_intervals": {
                "mean_finish": bootstrap_cluster_mean_ci(
                    place_clusters, resamples=bootstrap_resamples, seed=seed + index * 10,
                ),
                "playoff_rate": bootstrap_cluster_mean_ci(
                    playoff_clusters, resamples=bootstrap_resamples,
                    seed=seed + index * 10 + 1,
                ),
                "title_rate": bootstrap_cluster_mean_ci(
                    title_clusters, resamples=bootstrap_resamples,
                    seed=seed + index * 10 + 2,
                ),
            },
        }
        if fraction != 0:
            paired_place = []
            paired_playoff = []
            paired_title = []
            for current_runs, baseline_runs in zip(season_runs, baseline, strict=True):
                current_places = [float(run["regular_season_place"]) for run in current_runs]
                baseline_places = [float(run["regular_season_place"]) for run in baseline_runs]
                paired_place.append([a - b for a, b in zip(
                    current_places, baseline_places, strict=True,
                )])
                paired_playoff.append([
                    float(a <= 6) - float(b <= 6)
                    for a, b in zip(current_places, baseline_places, strict=True)
                ])
                paired_title.append([
                    float(a["we_won"]) - float(b["we_won"])
                    for a, b in zip(current_runs, baseline_runs, strict=True)
                ])
            row["paired_vs_naive"] = {
                "mean_finish_delta": bootstrap_cluster_mean_ci(
                    paired_place, resamples=bootstrap_resamples, seed=seed + index * 10 + 3,
                ),
                "playoff_rate_delta": bootstrap_cluster_mean_ci(
                    paired_playoff, resamples=bootstrap_resamples, seed=seed + index * 10 + 4,
                ),
                "title_rate_delta": bootstrap_cluster_mean_ci(
                    paired_title, resamples=bootstrap_resamples, seed=seed + index * 10 + 5,
                ),
            }
        fields.append(row)

    return {
        "seasons": seasons,
        "n_seasons": len(seasons),
        "n_slots_per_field": len(seasons) * teams,
        "field_strengths": fields,
        "bootstrap": {
            "method": "percentile_cluster",
            "sampling_unit": "nfl_season",
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "seed": seed,
        },
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--holdout", default="regression",
                        help="regression, locked, or an explicit season")
    parser.add_argument("--consume-locked-holdout", action="store_true",
                        help="explicitly consume the single-use locked evaluation")
    parser.add_argument("--force-locked-holdout", action="store_true",
                        help="reproduce a previously consumed locked evaluation")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="allow a run from an uncommitted working tree")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ffdata.experiments",
        description="Run reproducible, data-manifest-pinned experiments.",
    )
    p.add_argument("--root", type=Path, default=resolve_experiment_root(),
                   help="experiment root containing holdouts.json and results/")
    sub = p.add_subparsers(dest="command", required=True)

    weekly = sub.add_parser("weekly", help="weekly baseline vs LightGBM accuracy")
    _common(weekly)
    weekly.add_argument("--train-from", type=int, default=2019)
    weekly.add_argument("--positions", default="QB,RB,WR,TE")

    draft = sub.add_parser("draft", help="preseason projection rank accuracy")
    _common(draft)
    draft.add_argument("--scoring", choices=list(_RULES), default="ppr")
    draft.add_argument("--baseline", action="store_true",
                       help="evaluate the bare veteran model instead of the shipped realistic board")

    season = sub.add_parser("season", help="all-slot blind season simulation")
    _common(season)
    season.add_argument("--scoring", choices=list(_RULES), default="standard")
    season.add_argument("--teams", type=int, default=12)
    season.add_argument("--projector", choices=["gbm", "neural"], default="gbm")
    season.add_argument("--opponent", choices=["naive", "mixed", "sharp"], default="naive")
    season.add_argument("--sharp-fraction", type=float, default=0.5,
                        help="fraction of opponents that are sharp when --opponent mixed")
    season.add_argument("--noise", type=float, default=24.0)
    season.add_argument("--no-waivers", action="store_true")
    season.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    season.add_argument("--bootstrap-resamples", type=int, default=2000)

    sweep = sub.add_parser("season-sweep", help="multi-season mixed-field benchmark")
    _common(sweep)
    sweep.add_argument("--seasons", default="2022-2025")
    sweep.add_argument("--sharp-fractions", default="0,0.25,0.5,0.75,1")
    sweep.add_argument("--scoring", choices=list(_RULES), default="standard")
    sweep.add_argument("--teams", type=int, default=12)
    sweep.add_argument("--projector", choices=["gbm", "neural"], default="gbm")
    sweep.add_argument("--noise", type=float, default=24.0)
    sweep.add_argument("--no-waivers", action="store_true")
    sweep.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    sweep.add_argument("--bootstrap-resamples", type=int, default=2000)

    summary = sub.add_parser("summary", help="regenerate the Markdown results table")
    summary.add_argument("--output", type=Path)
    return p


def _run(args) -> tuple[dict, Path]:
    root = args.root
    results_dir = root / "results"
    policy = load_policy(root / "holdouts.json")
    season, tier = resolve_holdout(
        args.command, args.holdout, policy=policy, results_dir=results_dir,
        consume_locked=args.consume_locked_holdout,
        force_locked=args.force_locked_holdout,
    )
    git = git_metadata()
    if git["dirty"] and not args.allow_dirty:
        raise HoldoutError(
            "working tree is dirty; commit first or pass --allow-dirty for an exploratory run"
        )
    if args.command == "season-sweep":
        from .cli import parse_seasons
        seasons = parse_seasons(args.seasons)
        if max(seasons) != season:
            raise HoldoutError(
                f"season sweep ends at {max(seasons)}, but its selected holdout is {season}"
            )
        for candidate in seasons:
            _ensure_played(candidate)
    else:
        _ensure_played(season)

    if args.command == "weekly":
        positions = tuple(p.strip().upper() for p in args.positions.split(",") if p.strip())
        config = weekly_config(args.train_from, positions)
        runner = lambda: weekly_metrics(season, args.train_from, positions)
    elif args.command == "draft":
        realistic = not args.baseline
        config = {
            "scoring": args.scoring,
            "board": "realistic" if realistic else "baseline",
            "include_rookies": realistic,
            "career": realistic,
            "competition": realistic,
            "reconcile": realistic,
            "model_seed": 0,
        }
        runner = lambda: draft_metrics(season, args.scoring, realistic)
    elif args.command == "season":
        config = {
            "scoring": args.scoring, "teams": args.teams, "projector": args.projector,
            "waivers": not args.no_waivers, "opponent": args.opponent,
            "sharp_fraction": args.sharp_fraction, "noise": args.noise,
            "model_seed": 0, "bootstrap_seed": args.seed,
            "bootstrap_resamples": args.bootstrap_resamples,
        }
        runner = lambda: season_metrics(
            season, scoring=args.scoring, teams=args.teams, projector=args.projector,
            waivers=not args.no_waivers, opponent=args.opponent, noise=args.noise,
            sharp_fraction=args.sharp_fraction,
            seed=args.seed, bootstrap_resamples=args.bootstrap_resamples,
        )
    else:
        fractions = _parse_fractions(args.sharp_fractions)
        config = {
            "seasons": seasons, "sharp_fractions": fractions,
            "scoring": args.scoring, "teams": args.teams, "projector": args.projector,
            "waivers": not args.no_waivers, "opponent": "mixed", "noise": args.noise,
            "model_seed": 0, "bootstrap_seed": args.seed,
            "bootstrap_resamples": args.bootstrap_resamples,
        }
        runner = lambda: season_sweep_metrics(
            seasons, fractions, scoring=args.scoring, teams=args.teams,
            projector=args.projector, waivers=not args.no_waivers, noise=args.noise,
            seed=args.seed, bootstrap_resamples=args.bootstrap_resamples,
        )

    record = execute(
        runner, kind=args.command, season=season, tier=tier, config=config, clock=time.monotonic,
        git_reader=git_metadata, data_reader=data_metadata,
    )
    path = save_record(record, results_dir)
    render_summary(results_dir, root / "README.md")
    return record, path


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "summary":
        output = args.output or (args.root / "README.md")
        print(render_summary(args.root / "results", output), end="")
        return
    try:
        record, path = _run(args)
    except HoldoutError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(record["metrics"], indent=2, sort_keys=True))
    print(f"\nRecorded: {path}")
    print(f"Summary:  {args.root / 'README.md'}")


if __name__ == "__main__":
    main()
