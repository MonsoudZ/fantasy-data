"""Download, validation, atomic publishing, and season helpers (no real network)."""

import io
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

import ffdata.ingest as ingest
from ffdata.data import (
    DataValidationError,
    publish_parquet,
    register_cached_parquet,
    resolve_raw_path,
    validate_frame,
)
from ffdata.ingest import FIRST_SEASON, current_nfl_season


class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b"OK"


def test_download_retries_transient_errors_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("connection reset")
        return _FakeResp()

    monkeypatch.setattr(ingest.urllib.request, "urlopen", fake_urlopen)
    sleeps = []
    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)

    assert ingest._download("http://x", retries=3, backoff=1.0) == b"OK"
    assert calls["n"] == 3                      # failed twice, succeeded on the third
    assert sleeps == [1.0, 2.0]                 # exponential backoff between tries


def test_download_fails_fast_on_4xx(monkeypatch):
    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)

    monkeypatch.setattr(ingest.urllib.request, "urlopen", fake_urlopen)
    slept = {"n": 0}
    monkeypatch.setattr(ingest.time, "sleep", lambda s: slept.__setitem__("n", slept["n"] + 1))

    with pytest.raises(urllib.error.HTTPError):
        ingest._download("http://x", retries=3)
    assert slept["n"] == 0                       # a 404 is not retried


def test_download_gives_up_after_retries(monkeypatch):
    def fake_urlopen(req, timeout=0):
        raise TimeoutError("slow")

    monkeypatch.setattr(ingest.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    with pytest.raises(TimeoutError):
        ingest._download("http://x", retries=2)


def test_season_floor_and_rollover():
    import datetime as dt
    assert FIRST_SEASON == 2019
    assert current_nfl_season(dt.date(2025, 9, 1)) == 2025


def test_depth_charts_keep_latest_snapshot_per_team():
    """A single global max(dt) would drop every team whose latest chart predates
    another team's; keeping the latest PER TEAM preserves them all."""
    import pandas as pd

    from ffdata.ingest import _normalize_depth_charts
    df = pd.DataFrame({
        "team": ["BUF", "BUF", "KC", "KC"],
        "dt": ["2026-08-01", "2026-08-20", "2026-08-25", "2026-08-10"],
        "player": ["buf_old", "buf_new", "kc_new", "kc_old"],
    })
    out = _normalize_depth_charts(df, season=2026)
    assert set(zip(out["team"], out["player"])) == {("BUF", "buf_new"), ("KC", "kc_new")}
    assert (out["season"] == 2026).all()


def test_season_not_started_prefers_played_games_over_the_calendar():
    """The month-based rule wrongly calls a season 'started' in the ~week between
    the Sept 1 label rollover and real Week 1 kickoff. The data settles it."""
    import datetime as dt

    import duckdb

    from ffdata.ingest import season_not_started
    con = duckdb.connect()
    con.execute("create table weekly (season int, season_type varchar)")
    con.execute("insert into weekly values (2025, 'REG'), (2025, 'REG')")

    early_sept = dt.date(2026, 9, 3)          # after label rollover, before kickoff
    assert season_not_started(2025, early_sept, con=con) is False   # has games
    assert season_not_started(2026, early_sept, con=con) is True    # none yet
    # No connection -> the calendar heuristic still answers (pre-ingest CLI).
    assert season_not_started(2027, dt.date(2026, 7, 1)) is True


def test_upcoming_season_is_what_you_draft_for():
    """In the offseason `current` is the season already finished -- drafting
    against it would rank players for a season that's over."""
    import datetime as dt

    from ffdata.ingest import upcoming_nfl_season
    # Offseason: last completed is 2025, but you draft for 2026.
    assert current_nfl_season(dt.date(2026, 7, 21)) == 2025
    assert upcoming_nfl_season(dt.date(2026, 7, 21)) == 2026
    # Once games start, the season in progress is the one you're playing.
    assert upcoming_nfl_season(dt.date(2026, 10, 1)) == 2026
    # Just after a season ends, look ahead to the next one.
    assert upcoming_nfl_season(dt.date(2027, 2, 15)) == 2027


def test_data_path_precedence_and_installed_default(tmp_path):
    legacy = tmp_path / "checkout" / "data" / "raw"
    home = tmp_path / "home"
    assert resolve_raw_path({}, legacy_raw=legacy, user_home=home) == home / ".ff-data" / "raw"
    legacy.mkdir(parents=True)
    assert resolve_raw_path({}, legacy_raw=legacy, user_home=home) == legacy
    assert resolve_raw_path({"FFDATA_HOME": "/lake"}, legacy_raw=legacy) == Path("/lake/raw")
    assert resolve_raw_path(
        {"FFDATA_HOME": "/ignored", "FFDATA_DATA": "/exact/raw"}, legacy_raw=legacy
    ) == Path("/exact/raw")


def test_schema_validation_rejects_missing_empty_and_wrong_season():
    contract = {"required": {"season", "week"}, "any_of": ({"team", "club"},)}
    with pytest.raises(DataValidationError, match="empty"):
        validate_frame("weekly", pd.DataFrame(), contract, season=2026)
    with pytest.raises(DataValidationError, match="required columns"):
        validate_frame("weekly", pd.DataFrame({"season": [2026], "team": ["DEN"]}), contract)
    with pytest.raises(DataValidationError, match="expected one"):
        validate_frame("weekly", pd.DataFrame({"season": [2026], "week": [1]}), contract)
    with pytest.raises(DataValidationError, match="requested season 2026"):
        validate_frame(
            "weekly", pd.DataFrame({"season": [2025], "week": [1], "team": ["DEN"]}),
            contract, season=2026,
        )


def test_publish_is_atomic_and_records_provenance(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    dest = raw / "sample" / "sample_2026.parquet"
    frame = pd.DataFrame({"season": [2026], "week": [1], "value": [3.5]})
    publish_parquet(
        "sample", frame, dest, source="https://example.test/sample.parquet",
        contract={"required": {"season", "week"}}, season=2026, raw=raw,
    )
    assert pd.read_parquet(dest).to_dict("records") == frame.to_dict("records")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["files"]["sample/sample_2026.parquet"]
    assert entry["dataset"] == "sample"
    assert entry["season"] == 2026
    assert entry["source"] == "https://example.test/sample.parquet"
    assert entry["rows"] == 1 and len(entry["sha256"]) == 64

    original = dest.read_bytes()
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        publish_parquet("sample", frame, dest, source="new", raw=raw)
    assert dest.read_bytes() == original
    assert not list(dest.parent.glob("*.tmp"))


def test_fetch_validates_before_replacing_existing_file(tmp_path, monkeypatch):
    dest = tmp_path / "weekly_2026.parquet"
    dest.write_bytes(b"known-good")
    bad = pd.DataFrame({"season": [2026], "week": [1]})
    buf = io.BytesIO()
    bad.to_parquet(buf, index=False)
    monkeypatch.setattr(ingest, "_download", lambda url: buf.getvalue())
    with pytest.raises(DataValidationError, match="missing required columns"):
        ingest._fetch_to_parquet("weekly", "https://example.test/weekly.parquet", dest, season=2026)
    assert dest.read_bytes() == b"known-good"


def test_cached_parquet_is_validated_and_backfilled_once(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    dest = raw / "weekly" / "weekly_2025.parquet"
    dest.parent.mkdir(parents=True)
    pd.DataFrame({"season": [2025], "week": [1]}).to_parquet(dest, index=False)
    contract = {"required": {"season", "week"}}
    register_cached_parquet(
        "weekly", dest, source="https://example.test/weekly.parquet",
        contract=contract, season=2025, raw=raw,
    )
    manifest_path = tmp_path / "manifest.json"
    first = manifest_path.read_bytes()
    entry = json.loads(first)["files"]["weekly/weekly_2025.parquet"]
    assert entry["provenance"] == "backfilled_from_file_mtime"
    assert entry["rows"] == 1

    # A matching verified size/mtime/version entry is a no-op (and avoids
    # re-hashing a large historical lake on every routine ingest).
    monkeypatch.setattr("ffdata.data.pq.read_metadata", lambda path: pytest.fail("revalidated"))
    register_cached_parquet(
        "weekly", dest, source="https://example.test/weekly.parquet",
        contract=contract, season=2025, raw=raw,
    )
    assert manifest_path.read_bytes() == first
