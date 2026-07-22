"""Download retry/backoff + season helpers (no real network)."""

import urllib.error

import pytest

import ffdata.ingest as ingest
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
