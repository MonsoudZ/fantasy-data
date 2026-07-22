#!/usr/bin/env bash
# One-command onboarding for a fresh checkout (e.g. Claude Code cloud).
# Installs dependencies and rebuilds the data lake, which are gitignored.
#
#   bash scripts/setup.sh
#
# Needs: python3 and network access (nflverse pulls over HTTP).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"

echo "==> Installing dependencies"
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet -e ".[dev,web]"                     # core + tests/lint + web UI
$PY -m pip install --quiet -e ".[neural]" || echo "   (torch skipped — optional neural model)"

# Seasons are derived, not hardcoded, so this doesn't go stale each year.
UPCOMING=$($PY -c "from ffdata.ingest import upcoming_nfl_season; print(upcoming_nfl_season())")

echo "==> Ingesting nflverse data (through the last played season)"
$PY -m ffdata.cli                          # every source except pbp: weekly/injuries/snaps/
                                           # rosters/schedules/draft_picks/ngs/pfr
echo "==> Ingesting ${UPCOMING} preseason data (for drafting)"
# Only rosters exist before kickoff; schedules and draft_picks are all-years files
# already refreshed above, so the upcoming season's games + rookie class come free.
$PY -m ffdata.cli --datasets rosters --seasons "$UPCOMING"
# Optional (large, and only used by opt-in features):
#   $PY -m ffdata.cli --pbp

echo "==> Sanity check"
$PY -m pytest -q

cat <<'DONE'

Setup complete. Try:
  python -m ffdata.draft --season 2026        # draft board
  python -m ffdata.dynasty --season 2026       # dynasty values
  python -m ffdata.web                         # web UI at http://127.0.0.1:8000
DONE
