"""Local persistence for saved leagues (single-user, no server).

The web UI is otherwise stateless -- every visit re-enters season, scoring, team
count, and who's been drafted. This is a small JSON-backed store so a league can
be saved once and reloaded: name, season, scoring, teams, plus live draft state
(drafted players, keepers).

    from ffdata.store import League, save_league, list_leagues
    save_league(League(name="Home 12", season=2025, scoring="half", teams=12))
    [lg.name for lg in list_leagues()]

Storage is a single JSON file (atomic writes), defaulting to
``~/.ff-data/leagues.json`` and overridable with ``$FFDATA_STATE`` or an explicit
``path=`` -- the latter is what the tests use, so nothing here needs a data lake
or a network.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SCORING = ("ppr", "half", "standard")


@dataclass
class League:
    """A saved league configuration + its live draft state."""

    name: str
    season: int
    scoring: str = "ppr"
    teams: int = 12
    drafted: list[str] = field(default_factory=list)
    keepers: list = field(default_factory=list)   # list of [player, cost]

    def validated(self) -> "League":
        """Return self after checking fields; raise ValueError on bad input."""
        if not str(self.name).strip():
            raise ValueError("league name is required")
        if self.scoring not in _SCORING:
            raise ValueError(f"scoring must be one of {_SCORING}, got {self.scoring!r}")
        if not (2 <= int(self.teams) <= 32):
            raise ValueError(f"teams must be in 2..32, got {self.teams}")
        if not (1999 <= int(self.season) <= 2100):
            raise ValueError(f"season out of range: {self.season}")
        return self


def default_path() -> Path:
    """Where leagues live by default; override with $FFDATA_STATE."""
    env = os.environ.get("FFDATA_STATE")
    return Path(env) if env else Path.home() / ".ff-data" / "leagues.json"


def _load_raw(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_raw(path: Path, data: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: write a sibling temp file then replace, so a crash mid-write can't
    # corrupt the store or leave it half-written.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _key(name: str) -> str:
    return str(name).strip().lower()


def list_leagues(path: Path | None = None) -> list[League]:
    """All saved leagues, ordered by name."""
    path = path or default_path()
    raw = _load_raw(path)
    leagues = [League(**v) for v in raw.values()]
    return sorted(leagues, key=lambda lg: lg.name.lower())


def get_league(name: str, path: Path | None = None) -> League | None:
    """A saved league by name (case-insensitive), or None."""
    path = path or default_path()
    raw = _load_raw(path)
    row = raw.get(_key(name))
    return League(**row) if row else None


def save_league(league: League, path: Path | None = None) -> League:
    """Create or overwrite a league (keyed by case-insensitive name)."""
    path = path or default_path()
    league = league.validated()
    raw = _load_raw(path)
    raw[_key(league.name)] = asdict(league)
    _write_raw(path, raw)
    return league


def delete_league(name: str, path: Path | None = None) -> bool:
    """Remove a league; returns True if it existed."""
    path = path or default_path()
    raw = _load_raw(path)
    if _key(name) not in raw:
        return False
    del raw[_key(name)]
    _write_raw(path, raw)
    return True
