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

## Set your weekly lineup

```bash
python -m ffdata.optimize --week 15 --roster examples/my_roster.csv \
                          --opponent examples/opponent.csv --scoring ppr
```

Picks the lineup that maximizes your **win probability** against that opponent
(not just projected points), using the calibrated Monte Carlo with same-game
correlation. A roster file is one player name per line (see `examples/`). Add
`--projector neural` for the most accurate projections (slower); drop
`--opponent` to just get the highest-projected starters.

For large-field contests, `--mode tournament` maximizes the lineup's **ceiling**
(a high quantile of its own distribution) instead of beating one opponent —
`--ceiling 0.97` targets the top ~3%. Pure ceiling-optimization stacks only
marginally, though (the +0.20 QB-receiver correlation is diluted across an
8-player lineup while the projection cost of stacking is direct).

`--mode stack` is the DFS-style answer: it builds the best-ceiling lineup
*around a game stack* — a QB + `--stack-size` of his receivers + `--bringback`
opponent receivers — concentrating several correlations so the fat right tail
survives. This is how real DFS optimizers enforce stack rules.

### Web UI

```bash
pip install fastapi "uvicorn[standard]"
python -m ffdata.web        # -> http://127.0.0.1:8000
```

A polished browser front-end over all of the above: pick week/scoring/mode,
search the full slate of ~300 projectable players (or paste your roster), and
get the recommended lineup with its win probability or ceiling. "Full slate"
optimizes over every player (DFS); the first run for a scoring type trains the
model (~2 min), then it's instant.

## Find player-prop edges

```bash
python -m ffdata.props --week 15 --props examples/props.csv
```

Prices a table of prop lines (`player,market,line,over_odds,under_odds`) against
per-stat projections and reports the **+EV** bets. Each market
(passing/receiving/rushing yards, receptions, passing TDs) has its own model;
P(over) comes from out-of-sample residuals, **validated calibrated to nominal
within ~2-3 points cross-season** — so the edges are honest. nflverse ships no
prop odds, so you supply the lines (export or type them in); the engine and its
calibration are what this provides.

## Roadmap

1. ~~Historical dataset + ingestion~~ (this repo)
2. ~~Feature layer: rolling usage, opponent-adjusted defense, Vegas implied totals~~
   (`ffdata/features.py` — leak-free player-week table via `build_features()`.
   Also models trailing snap share (via the rosters gsis↔pfr crosswalk) and
   current-week injury-report status; adding them lifts projection MAE, RMSE,
   and weekly rank on 2024.)
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
6. ~~Neural projector + win-probability lineup optimizer~~
   (`ffdata/neural.py` — a GRU over player trajectories, the most accurate
   single model (beats LightGBM on MAE/RMSE/rank across 2023-25); promoted to
   the default projector in `matchup.py`. `ffdata/optimize.py` — picks the
   roster that maximizes win probability, not projected points, using the
   calibrated Monte Carlo.)
7. ~~Same-game correlation~~
   (`ffdata/correlation.py` — a Gaussian copula over same-game players. The
   QB<->own-receiver "stack" residual correlation is +0.20 in the data, so
   independent sampling understated a stack's variance by ~30%. The copula
   restores it without shifting any player's calibrated marginal.)
8. ~~Player-prop edge finder~~
   (`ffdata/props.py` — a per-stat model per market (passing/receiving/rushing
   yards, receptions, passing TDs); P(over) from out-of-sample residuals,
   validated calibrated to nominal within ~2-3 pts cross-season. Prices
   user-supplied prop lines into +EV bets. Props are softer than game lines, so
   this is the market our projections might actually beat — but you bring the
   odds; nflverse has none.)

**What we learned:** across six independent tests — a neural model, a stacked
ensemble, and every rich data source (Next Gen Stats, PFR advanced, play-by-play
red-zone, opponent matchup) — the weekly point-projection error floor (~±6 RMSE)
did not move. It's dominated by irreducible game-day variance; the predictable
signal is already captured by usage, efficiency, and the Vegas line. The payoff
therefore shifts from *predicting* better to *deciding* better under calibrated
uncertainty (steps 4 and 6). Those experiments live behind opt-in flags in
`features.py` (`include_ngs` / `include_extra` / `include_pbp` / `include_matchup`).
