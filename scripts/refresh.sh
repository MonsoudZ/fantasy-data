#!/usr/bin/env bash
# Refresh the DRAFT-relevant data right before your draft, so the board reflects
# the latest roster moves, injuries, and rookie landing spots.
#
#   bash scripts/refresh.sh
#
# Run it a day or two before the draft. It force re-downloads the sources a draft
# actually reads (rosters, injuries, schedules, draft picks, last-season stats) and
# refreshes the live Sleeper availability feed. The season GBM/board rebuild
# themselves off the fresh files on the next `python -m ffdata.web`.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
UPCOMING=$($PY -c "from ffdata.ingest import upcoming_nfl_season; print(upcoming_nfl_season())")

echo "==> Force-refreshing draft sources through the last played season"
# rosters/injuries feed availability + the room; schedules feeds SOS + game lines;
# draft_picks feeds the rookie model; weekly/snaps feed the projection.
$PY -m ffdata.cli --force \
    --datasets rosters injuries schedules draft_picks weekly snap_counts

echo "==> Force-refreshing ${UPCOMING} preseason rosters (draft/keeper/rookie data)"
$PY -m ffdata.cli --force --datasets rosters --seasons "$UPCOMING"

echo "==> Refreshing live availability (Sleeper: today's IR / PUP / suspensions)"
$PY -m ffdata.cli --live --force || echo "   (Sleeper refresh skipped — non-fatal)"

cat <<DONE

Data refreshed for ${UPCOMING}.

  * Consensus ADP is NOT auto-pulled (it comes from a rankings PDF). To update it,
    download a fresh sheet and run:
        python scripts/ingest_adp.py ~/Downloads/<rankings>.pdf --source CBS
  * The "no 2026 team" (warn) flags on the board shrink as rosters fill in -- if a
    star still shows one after this, verify his team manually.

Restart the web UI to pick it all up:  python -m ffdata.web
DONE
