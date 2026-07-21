# ff-data

[![CI](https://github.com/MonsoudZ/fantasy-data/actions/workflows/ci.yml/badge.svg)](https://github.com/MonsoudZ/fantasy-data/actions/workflows/ci.yml)

Fantasy football data lake + platform-agnostic scoring. Foundation for
projections, lineup optimization, and edge-finding vs Vegas lines.

## Design

- **Raw stats are the source of truth.** All data pulled directly from
  [nflverse](https://github.com/nflverse/nflverse-data) GitHub release assets —
  no API keys, no wrapper library, updated nightly in-season.
- **Scoring is config, not data.** You play on multiple platforms, so fantasy
  points are computed from raw stats per `ScoringRules`. One dataset scores
  PPR, half-PPR, TE-premium, or any custom league identically.
- **Parquet + DuckDB, no database server.** Files land in `data/raw/`,
  DuckDB queries them in place via views.

## Setup

```bash
pip install -r requirements.txt
```

The projection and edge models use LightGBM, which needs an OpenMP runtime:

```bash
brew install libomp          # macOS
sudo apt-get install libgomp1  # Debian/Ubuntu
```

## Ingest

```bash
python -m ffdata.cli                          # 2018-present, all core datasets
python -m ffdata.cli --seasons 2015-2024      # more history
python -m ffdata.cli --pbp                    # include play-by-play (large)
python -m ffdata.cli --force                  # re-download everything
```

Idempotent: cached seasons are skipped; the current season always refreshes.

## Test

```bash
pip install -r requirements-dev.txt
pytest
```

Unit tests are synthetic and need no data (scoring exactness, leak-free
features, de-vig/payout math). The integration tests validate `score()` against
nflverse's precomputed columns on the real lake, and skip automatically until
you've ingested. CI runs the unit tests on every push.

## Datasets

| view          | grain                | why you care |
|---------------|----------------------|--------------|
| `weekly`      | player x week        | the modeling target: volume, efficiency, target share, air yards |
| `schedules`   | game                 | Vegas spread/total/moneyline, rest days, roof/surface |
| `injuries`    | player x week report | practice + game status |
| `snap_counts` | player x game        | opportunity share — leading indicator of usage |
| `rosters`     | player x week        | team/position/depth/status joins |
| `pbp`         | play                 | red-zone touches, EPA splits, situational features |

## Query

```python
from ffdata.db import connect
from ffdata.scoring import score, PPR, HALF_PPR, ScoringRules

con = connect()
weekly = con.sql("select * from weekly where season = 2024").df()
scored = score(weekly, HALF_PPR)

# custom league: TE premium, 6-pt passing TDs
my_league = ScoringRules(pass_td=6.0, te_reception_bonus=0.5)
```

Validated: `score(weekly, PPR)` matches nflverse's precomputed
`fantasy_points_ppr` with 0.00 max deviation on the 2024 season.

## Roadmap

1. ~~Historical dataset + ingestion~~ (this repo)
2. ~~Feature layer: rolling usage, opponent-adjusted defense, Vegas implied totals~~
   (`ffdata/features.py` — leak-free player-week table via `build_features()`)
3. ~~Projection model (LightGBM) vs trailing-average baseline~~
   (`ffdata/projections.py` — walk-forward backtest; LightGBM beats the
   trailing-average baseline on MAE, RMSE, and weekly rank on 2023–24)
4. ~~Matchup win probability via Monte Carlo over projections~~
   (`ffdata/matchup.py` — resamples out-of-sample residuals; predictive
   intervals calibrated to ~1pt out-of-sample on 2024)
5. ~~Edge finder: model probability vs implied odds, tracked over time~~
   (`ffdata/edge.py` — walk-forward game models vs de-vigged lines, with a
   bet-the-edge backtest. Honest finding: a fundamentals-only model from public
   nflverse data lands ~0.6-1.1 pts MAE short of the closing line, so no edge
   survives the vig on 2023-24 game markets — spread/total ROI ≈ break-even,
   moneyline negative. The harness is built to test a *better* signal next.)
