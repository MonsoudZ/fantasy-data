"""Experiment policy, reproducibility metadata, intervals, and artifact rendering."""

import json

import pytest

import ffdata.experiments as experiments
from ffdata.experiment_registry import (
    DEFAULT_POLICY,
    HoldoutError,
    bootstrap_cluster_mean_ci,
    bootstrap_mean_ci,
    build_record,
    data_metadata,
    render_summary,
    resolve_experiment_root,
    resolve_holdout,
    save_record,
)


def test_experiment_root_override_wins(tmp_path):
    assert resolve_experiment_root({"FFDATA_EXPERIMENTS": str(tmp_path)}) == tmp_path


def test_locked_holdout_requires_explicit_single_use(tmp_path):
    results = tmp_path / "results"
    with pytest.raises(HoldoutError, match="--consume-locked-holdout"):
        resolve_holdout(
            "weekly", "locked", policy=DEFAULT_POLICY, results_dir=results,
        )
    assert resolve_holdout(
        "weekly", "locked", policy=DEFAULT_POLICY, results_dir=results,
        consume_locked=True,
    ) == (2026, "locked")

    record = build_record(
        "weekly", 2026, "locked", {}, {"models": {}}, duration_seconds=1,
        git={"commit": "abc", "dirty": False}, data={"available": True},
    )
    save_record(record, results)
    with pytest.raises(HoldoutError, match="already consumed"):
        resolve_holdout(
            "weekly", "locked", policy=DEFAULT_POLICY, results_dir=results,
            consume_locked=True,
        )
    assert resolve_holdout(
        "weekly", "locked", policy=DEFAULT_POLICY, results_dir=results,
        consume_locked=True, force_locked=True,
    ) == (2026, "locked")


def test_explicit_seasons_are_labeled_by_policy(tmp_path):
    results = tmp_path / "results"
    assert resolve_holdout(
        "draft", "2025", policy=DEFAULT_POLICY, results_dir=results,
    ) == (2025, "regression")
    assert resolve_holdout(
        "draft", "2024", policy=DEFAULT_POLICY, results_dir=results,
    ) == (2024, "exploratory")


def test_bootstrap_intervals_are_deterministic_and_honest():
    a = bootstrap_mean_ci([1, 2, 3, 4], resamples=500, seed=7)
    b = bootstrap_mean_ci([1, 2, 3, 4], resamples=500, seed=7)
    assert a == b
    assert a["estimate"] == 2.5 and a["low"] <= 2.5 <= a["high"]
    constant = bootstrap_mean_ci([1, 1, 1], resamples=20)
    assert constant["low"] == constant["high"] == 1.0


def test_cluster_bootstrap_resamples_whole_seasons_deterministically():
    clusters = [[1, 3], [5, 7]]
    a = bootstrap_cluster_mean_ci(clusters, resamples=500, seed=9)
    b = bootstrap_cluster_mean_ci(clusters, resamples=500, seed=9)
    assert a == b
    assert a["estimate"] == 4.0
    assert a["n_clusters"] == 2 and a["n"] == 4
    assert a["low"] <= a["estimate"] <= a["high"]


def test_data_metadata_fingerprints_manifest_and_assets(tmp_path):
    manifest = {
        "manifest_version": 1,
        "files": {"weekly/a.parquet": {"sha256": "a" * 64}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    meta = data_metadata(path)
    assert meta["available"] and meta["asset_count"] == 1
    assert len(meta["manifest_sha256"]) == 64
    assert len(meta["asset_set_sha256"]) == 64
    assert meta["assets"]["weekly/a.parquet"] == "a" * 64


def test_record_save_and_markdown_summary(tmp_path):
    record = build_record(
        "season", 2025, "regression", {"opponent": "naive"},
        {
            "confidence_intervals": {
                "mean_finish": {"estimate": 4.5, "low": 3.5, "high": 5.5},
                "playoff_rate": {"estimate": 0.75, "low": 0.5, "high": 0.9},
                "title_rate": {"estimate": 0.25, "low": 0.05, "high": 0.5},
            }
        },
        duration_seconds=12.3456,
        git={"commit": "abcdef123456", "dirty": False}, data={"available": True},
    )
    path = save_record(record, tmp_path / "results")
    loaded = json.loads(path.read_text())
    assert loaded["duration_seconds"] == 12.346
    summary_path = tmp_path / "README.md"
    text = render_summary(tmp_path / "results", summary_path)
    assert "season" in text and "naive" in text and "2025 (regression)" in text
    assert "playoffs 75.0%" in text and "`abcdef12`" in text
    assert summary_path.read_text() == text


def test_weekly_cli_run_records_json_without_consuming_locked(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "_ensure_played", lambda season: None)
    monkeypatch.setattr(
        experiments, "git_metadata",
        lambda: {"commit": "abc", "branch": "main", "dirty": False, "dirty_paths": []},
    )
    monkeypatch.setattr(
        experiments, "data_metadata",
        lambda: {"available": True, "asset_set_sha256": "data"},
    )
    monkeypatch.setattr(
        experiments, "weekly_metrics",
        lambda season, train_from, positions: {
            "models": {"lightgbm": {"MAE": 4.2, "RMSE": 6.1, "weekly_spearman": 0.55}}
        },
    )
    monkeypatch.setattr(
        experiments, "weekly_config",
        lambda train_from, positions: {"train_from": train_from, "positions": positions},
    )
    args = experiments.parser().parse_args([
        "--root", str(tmp_path), "weekly", "--holdout", "regression",
    ])
    record, path = experiments._run(args)
    assert record["holdout"] == {"season": 2025, "tier": "regression"}
    assert path.exists()
    assert (tmp_path / "README.md").exists()


def test_cli_rejects_dirty_tree_without_override(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiments, "git_metadata",
        lambda: {"commit": "abc", "branch": "main", "dirty": True,
                 "dirty_paths": ["ffdata/model.py"]},
    )
    args = experiments.parser().parse_args(["--root", str(tmp_path), "draft"])
    with pytest.raises(HoldoutError, match="working tree is dirty"):
        experiments._run(args)


def test_execute_rejects_data_changes_during_run():
    from ffdata.experiment_registry import execute

    states = iter([
        {"available": True, "asset_set_sha256": "before"},
        {"available": True, "asset_set_sha256": "after"},
    ])
    with pytest.raises(RuntimeError, match="data manifest changed"):
        execute(
            lambda: {"MAE": 1.0}, kind="weekly", season=2025, tier="regression",
            config={}, clock=lambda: 1.0,
            git_reader=lambda: {"commit": "abc", "dirty": False},
            data_reader=lambda: next(states),
        )


def test_execute_requires_provenance_manifest():
    from ffdata.experiment_registry import execute

    with pytest.raises(HoldoutError, match="no provenance manifest"):
        execute(
            lambda: pytest.fail("runner should not start"),
            kind="draft", season=2025, tier="regression", config={}, clock=lambda: 1.0,
            git_reader=lambda: {"commit": "abc", "dirty": False},
            data_reader=lambda: {"available": False},
        )


def test_season_sweep_reuses_context_and_reports_paired_clustered_results(monkeypatch):
    import ffdata.season_sim as season_sim

    prepared = []
    observed_contexts = []

    def fake_prepare(season, **kwargs):
        ctx = {"season": season}
        prepared.append(ctx)
        return ctx

    def fake_run_all_slots(season, **kwargs):
        observed_contexts.append(kwargs["ctx"])
        fraction = kwargs["sharp_fraction"]
        places = [1, 2] if fraction == 0 else [2, 2]
        return {
            "runs": [
                {"regular_season_place": place, "we_won": fraction == 0 and slot == 0}
                for slot, place in enumerate(places)
            ]
        }

    monkeypatch.setattr(season_sim, "prepare", fake_prepare)
    monkeypatch.setattr(season_sim, "run_all_slots", fake_run_all_slots)
    result = experiments.season_sweep_metrics(
        [2024, 2025], [0.0, 1.0], teams=2, bootstrap_resamples=100,
    )

    assert len(prepared) == 2
    assert observed_contexts == [prepared[0], prepared[0], prepared[1], prepared[1]]
    assert result["bootstrap"]["sampling_unit"] == "nfl_season"
    assert result["field_strengths"][0]["mean_finish"] == 1.5
    sharp = result["field_strengths"][1]
    assert sharp["mean_finish"] == 2.0
    assert sharp["paired_vs_naive"]["mean_finish_delta"]["estimate"] == 0.5
    assert sharp["confidence_intervals"]["mean_finish"]["n_clusters"] == 2


def test_season_sweep_parser_defaults_to_regression_window():
    args = experiments.parser().parse_args(["season-sweep"])
    assert args.seasons == "2022-2025"
    assert experiments._parse_fractions(args.sharp_fractions) == [0, 0.25, 0.5, 0.75, 1]
