# CLAUDE.md — ff-data

Fantasy football data platform. Raw nflverse stats → league-agnostic scoring →
projections (weekly + season) → decision tools (lineup optimizer, draft board,
dynasty, prop edges) → a tabbed web UI.

## First-time setup (venv + data are gitignored)

```bash
bash scripts/setup.sh          # installs deps + ingests data + runs tests
```

Or manually: `pip install -r requirements-dev.txt` (add `torch` for the neural
model, `fastapi "uvicorn[standard]"` for the web UI), then
`python -m ffdata.cli --seasons 2019-2025` and
`python -m ffdata.cli --datasets rosters --seasons 2026` (for a 2026 draft).
Network access required (nflverse pulls over HTTP).

## Layout (`ffdata/`)

| module | what |
|---|---|
| `sources.py` `ingest.py` `db.py` | download nflverse parquet → DuckDB views |
| `scoring.py` | fantasy points from raw stats per `ScoringRules` (PPR/half/std/custom) |
| `features.py` | leak-free weekly modeling table (`build_features`); opt-in flags for ngs/pfr/pbp/matchup |
| `projections.py` | weekly GBM vs trailing baseline, walk-forward backtest |
| `neural.py` | GRU sequence projector (`NeuralProjector`); needs torch (lazy-imported) |
| `matchup.py` | Monte Carlo lineup win-prob; residual sampler |
| `correlation.py` | Gaussian copula for same-game correlation |
| `optimize.py` | lineup optimizer (h2h / tournament / stack), superflex + DEF/K slots, free-agent finder + weekly CLI |
| `kdst.py` | kicker + team-defense (DST) scoring & leak-free trailing projections |
| `betting.py` | American-odds / de-vig math + empirical P(over), shared by props/gamelines |
| `props.py` | player-prop edge finder (per-stat models; you supply odds) |
| `gamelines.py` | game total/spread/moneyline forecast vs market (informational; lines from `schedules`) |
| `draft.py` | preseason season projections, VOR, snake/auction, keepers, trades, rookies |
| `dynasty.py` | age curves (delta method) + multi-year dynasty value |
| `store.py` | JSON persistence for saved leagues + lineup teams (incl. custom scoring) |
| `sleeper.py` | import a league from Sleeper's public API (settings, scoring, roster, draft) |
| `advice.py` | grounded Claude explanations of compare/keeper/trade decisions (opt-in) |
| `web.py` `static/index.html` | FastAPI + tabbed UI |

## Common commands

```bash
pytest                                              # 55 tests; integration tests skip w/o data
python -m ffdata.optimize --week 15 --roster r.csv --opponent o.csv
python -m ffdata.draft --season 2026                # draft board (VOR + auction $)
python -m ffdata.dynasty --season 2026
python -m ffdata.props --week 15 --props lines.csv
python -m ffdata.web                                # http://127.0.0.1:8000
```

## Conventions & guardrails

- **Everything is measured, not asserted.** Each model/feature has a backtest;
  keep that discipline — validate leak-free and report honestly, including
  negatives. The git log is a research notebook.
- **Leakage is the cardinal sin.** Weekly features are trailing/shifted; draft
  features use only prior-season data + preseason-known context (schedule, age).
- **Two projection regimes:** weekly (in-season, trailing features) and season
  (preseason, prior-year features). They are *different models* — don't conflate.

## Findings already established (don't re-litigate)

- The weekly point-projection **error floor (~±6 RMSE) is irreducible**. Confirmed
  6 ways (neural, ensemble, NGS, PFR+weather, pbp red-zone, opponent matchup).
  NGS/PFR/pbp/matchup features are opt-in and OFF by default because they don't help.
- The **neural GRU** beats the GBM on rank across 2023-25 but errs ~0.97-correlated
  with it — it's the default projector in `matchup.py`.
- Monte Carlo intervals are **calibrated to ~1pt** out of sample; prop P(over) to
  ~2-3pt. Same-game QB↔receiver residual correlation is **+0.20**.
- **Game betting markets are efficient** to a public-data model (no edge survives
  the vig). The bet-tracking *edge finder* was pruned; the game models live on in
  `gamelines.py` as an **informational forecast-vs-market view** (totals/spread/
  moneyline, lines straight from `schedules`) — a sanity check on the line, not a
  profitable edge. Reusable odds math is in `betting.py`. Props *might* be
  beatable but need a real odds source (nflverse has none).
- The **stacked ensemble ("colony")** was a dead-end too — stacking can't beat the
  irreducible floor when models err ~0.97-correlated — and was **removed**. Finding
  kept; code gone.
- Draft: the season GBM alone loses to naive "last year's points"; the shipped
  projection is a **0.4-model / 0.6-prior blend** (rank ~0.72). Delta-method age
  curves: RB peaks ~24 (cliff), WR ~25, TE ages gracefully.
- Same-game correlation and stacking are **real but modest** — stacking is an
  ownership/leverage play, not a raw-ceiling win (we have no ownership data).

## Data notes

- `data/` and `.venv/` are gitignored. Seasons 2019-2025 for weekly/injuries/
  snaps/rosters; schedules is one all-seasons file (1999-2026); 2026 rosters are
  preseason (for drafting). `pbp` is opt-in and large.
- Draft/dynasty values **honor any `ScoringRules`** (scored from raw stats via
  `scoring.score()`, same as the weekly path); default PPR. CLIs take
  `--scoring ppr|half|standard`; the API takes a `rules=` / `scoring` arg.
- Rookies: a **draft-capital model** (`draft.rookie_projection`, needs the
  `draft_picks` source) projects rookie-season points from where a player was
  drafted and folds them into `draft_board` (`include_rookies=True`). ⚠️ Scaffolded
  but **not yet backtested on real data** — run `draft.backtest_rookies()` before
  trusting the magnitudes. Degrades to veterans-only if `draft_picks` isn't ingested.
- **Grounded advice** (`advice.py`, "Explain — why?" buttons on the draft tab's
  compare/keeper/trade results): asks Claude (`claude-opus-4-8`, adaptive thinking)
  to explain a decision, but **grounded** — the system prompt forbids any stat not
  in the `facts` dict, which is the engine's own output (proj/VOR/auction/rank +
  keeper surplus / trade totals + the league's scoring). So it phrases and weighs
  the trade-offs the numbers imply; it can't invent a projection. Optional extra
  (`pip install '.[advice]'`), needs `ANTHROPIC_API_KEY`; `advice.available()`
  gates it and `/api/config` exposes the flag so the UI only shows the button when
  it's on. The endpoint (`/api/advice`, dispatch on `kind`) reuses the same board +
  `keeper_value`/`trade_value`/compare-rows the tools do, so the explanation and
  the table can never disagree. ⚠️ Prompt assembly + the availability gate are
  unit-tested with a mocked client; the **live API path is unvalidated** (no egress
  when built) — confirm once you set a key.
- **Kicker + team defense (K/DST)** (`kdst.py`): standard leagues start a K and a
  DEF (QB/RB/RB/WR/WR/TE/FLEX/DEF/K), so the app scores and projects them.
  `ScoringRules` gained kicker (distance-laddered FG + PAT + miss) and DST (sack/
  int/fumble/TD/safety/block + a fixed standard points-allowed tier ladder) fields;
  `score_kicker`/`score_dst` compute them from raw stats (graceful columns).
  `project_kdst(season, week, rules)` returns K + DEF board rows via a **trailing
  average** — the honest model for these near-irreducible positions — leak-free
  (only prior weeks feed the mean), and `web._board` appends them so the optimizer/
  free-agent finder can fill the DEF/K slots. Sleeper import now maps K/DEF starter
  slots + roster (defense stored as `<TEAM> DST` to match the board). ⚠️ Two
  validation gaps, flagged in the module: kicker distance-bucket **column names**
  vary by nflverse schema era (falls back to flat `fg_made`), and DST **counting
  stats** (sacks/takeaways/def TDs) need a defensive box-score source this project
  doesn't ingest yet — so DST is points-allowed-dominated. The scoring math + leak-
  free trailing are unit-tested; **magnitudes are UNVALIDATED** (no lake here).
  Draft-board K/DEF ranking is deliberately **out of scope** (you stream them; VOR
  is ~flat) — this is a weekly-lineup feature.
- **Superflex weekly slots** (`optimize.py`): `slots_from_lineup(lineup)` turns a
  `{starters, flex, superflex}` config into the optimizer's slot tuple, adding a
  `SUPERFLEX` slot (QB-eligible) so a 2-QB league optimizes its *real* lineup —
  a second QB can now start. Threads through `/api/optimize` (and the opponent's
  assembled lineup) via `OptRequest.lineup`; the lineup tab has a **Superflex /
  2-QB** toggle that sends the canonical superflex config. 1-QB leagues are
  unaffected (default slots).
- **Free-agent / waiver finder** (`optimize.free_agent_advice`, `/api/freeagents`,
  lineup tab): ranks available players by **marginal starting-lineup gain**, not
  raw projection — for each free agent it recomputes your best starting lineup
  with him added and reports the point gain over your current best (0 if he
  doesn't crack it), naming the starter he'd bench. Superflex-aware (same slots),
  honors scoring, and takes an optional `exclude` list (players rostered by
  others). This is the honest season-long-pickup metric; it's projection-based,
  *not* the Monte Carlo win-prob objective (that answers "win this one matchup").
  A grounded "Explain" button isn't wired here yet — that'd need `/api/advice` to
  recompute free-agent facts server-side (it only carries board config today), so
  the ranked table stands on its own for now.
- **Sleeper import** (`sleeper.py`, web tab): pulls a league by username via
  Sleeper's public read-only API (no auth) → saves a `store.League` (settings,
  exact custom scoring, drafted, starting lineup) + a `store.Team` (your roster).
  Custom scoring is a full `ScoringRules` (stored `rules` dict; label `custom`)
  and `roster_positions` becomes a `lineup` dict `{starters, flex, superflex}` so
  VOR is superflex-aware (`_replacement_ranks` deepens QB for SF slots) — both
  thread through the draft/lineup endpoints. ⚠️ The pure mappers are unit-tested;
  the live HTTP path is **unvalidated** (egress was blocked when built) — confirm
  against a real account. ESPN/Yahoo are not built (unofficial-cookie / OAuth).
