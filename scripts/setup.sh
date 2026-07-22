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

echo "==> Ingesting nflverse data (2019-2025 core + 2026 rosters)"
$PY -m ffdata.cli --seasons 2019-2025                          # weekly/injuries/snaps/rosters/schedules
$PY -m ffdata.cli --datasets rosters --seasons 2026            # preseason rosters for a 2026 draft
# Optional extras (bigger / opt-in features):
#   $PY -m ffdata.cli --pbp --seasons 2019-2025                # play-by-play (large)
#   $PY -m ffdata.cli --datasets ngs_receiving ngs_passing ngs_rushing pfr_pass pfr_rec pfr_rush

echo "==> Sanity check"
$PY -m pytest -q

cat <<'DONE'

Setup complete. Try:
  python -m ffdata.draft --season 2026        # draft board
  python -m ffdata.dynasty --season 2026       # dynasty values
  python -m ffdata.web                         # web UI at http://127.0.0.1:8000
DONE
