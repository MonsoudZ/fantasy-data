"""Web-layer input validation, cache bounding, and prop parsing.

FastAPI/pydantic are optional deps (the `web` extra), so these skip cleanly in
CI where they aren't installed -- same pattern as the data-lake integration
tests. They guard the hardening added after the audit: bounded request fields
(no `range(2019, 2_000_000_000)` blowups) and a capped per-config cache.
"""

import pytest

pytest.importorskip("fastapi")

from pydantic import ValidationError  # noqa: E402

from ffdata.web import (  # noqa: E402
    DraftRequest, OptRequest, PropsRequest, _cache_put, _MAX_CACHE, _parse_props,
)


def test_optrequest_defaults_to_a_sane_season():
    r = OptRequest(week=5)
    assert 1999 <= r.season <= 2100


@pytest.mark.parametrize("kwargs", [
    {"week": 5, "season": 2_000_000_000},   # the range()-blowup vector
    {"week": 99},                            # week out of range
    {"week": 0},
    {"week": 5, "ceiling": 1.5},             # quantile must be < 1
])
def test_optrequest_rejects_out_of_range(kwargs):
    with pytest.raises(ValidationError):
        OptRequest(**kwargs)


def test_draft_and_props_requests_bound_their_fields():
    with pytest.raises(ValidationError):
        DraftRequest(teams=999)
    with pytest.raises(ValidationError):
        DraftRequest(n=100_000)
    with pytest.raises(ValidationError):
        PropsRequest(week=50)


def test_cache_put_caps_size_and_evicts_oldest():
    cache: dict = {}
    for i in range(_MAX_CACHE + 5):
        _cache_put(cache, i, i)
    assert len(cache) == _MAX_CACHE
    assert 0 not in cache                    # oldest evicted
    assert (_MAX_CACHE + 4) in cache         # newest kept
    # Re-putting an existing key doesn't grow or evict.
    existing = next(iter(cache))
    _cache_put(cache, existing, "v")
    assert len(cache) == _MAX_CACHE and cache[existing] == "v"


def test_parse_props_skips_header_and_malformed_rows():
    text = ("player,market,line,over,under\n"
            "Josh Allen,passing_yards,250.5,-110,-110\n"
            "bad,row\n")
    df = _parse_props(text)
    assert list(df["player"]) == ["Josh Allen"]
    assert df.iloc[0]["line"] == 250.5
