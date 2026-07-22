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
| `backtest_draft.py` | retrospective draft-and-win backtest: draft blind, replay the real season |
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

- `data/` and `.venv/` are gitignored. Per-season files (weekly/injuries/snaps/
  rosters/pfr) exist only for **played** seasons; all-years files (schedules,
  draft_picks, ngs) already cover the future. So the preseason lake is: played
  seasons through `current_nfl_season()`, plus **rosters for the upcoming
  season** — which is all a draft needs (the schedule and the rookie class come
  from the all-years files). `scripts/setup.sh` derives both, so it never goes
  stale. `pbp` is opt-in and large.
- **`current_nfl_season()` vs `upcoming_nfl_season()`** (`ingest.py`): the first
  is the most recently *played* season, the second is the one you **draft for**.
  In the offseason they differ, so draft/dynasty CLIs and the draft board default
  to *upcoming* — defaulting to `current` would rank players for a season that's
  already over. In-season tools (lineup, props, game lines) use `current`.
- `weekly` keeps skill positions **plus K** — `kdst.build_kicker` reads kickers
  out of it, so filtering them at ingest silently kills kicker projections.
  Team defense comes from `schedules`, not `weekly`.
- Draft/dynasty values **honor any `ScoringRules`** (scored from raw stats via
  `scoring.score()`, same as the weekly path); default PPR. CLIs take
  `--scoring ppr|half|standard`; the API takes a `rules=` / `scoring` arg.
- Rookies: a **draft-capital model** (`draft.rookie_projection`, needs the
  `draft_picks` source) projects rookie-season points from where a player was
  drafted and folds them into `draft_board` (`include_rookies=True`).
  **Backtested (2022-25)**: draft pick is nearly the whole signal — naive pick
  order ranks 0.575, the original multi-feature GBM only 0.510 (it overfit ~350
  rows). Ships as a **monotone pick-only curve**: 0.566, matching the naive
  ordering while still emitting the points VOR/auction need. Position is
  deliberately excluded (as features 0.510, as a per-position scale 0.520 — both
  worse). Expect ~0.57 rank / ~45 pts MAE: rookie values are a **prior, not a
  projection**, and the curve is stepped, so ties are real (broken by pick).
  Degrades to veterans-only if `draft_picks` isn't ingested.
- **Rookie opportunity is context, not a feature** (`draft.rookie_context`): the
  drafting team's vacated vs returning production at that position, plus the
  preseason depth-chart rank (`depth_charts` source). Tested as model features
  and they made ranking *worse* every year (0.57 → 0.51 raw, 0.54 even with
  domain-correct monotone constraints). Why: the signals are real but weak
  (vacated +0.14, returning −0.09 vs **pick +0.62**), teams already draft partly
  for need (QB +0.31, TE +0.23 corr between vacated share and an earlier pick),
  and ~350 training rookies can't afford the variance. So it's surfaced for a
  human to weigh, and shown under each rookie on the draft-board UI (an `R`
  badge plus a situation line). Summed vacated points alone mislead, so the
  context names **who is still ahead of him**, his **depth-chart rank**, and the
  team's **pass rate** — scheme caps the pie. 2026 is the case in point: Makai
  Lemon (pick 20) has 273 vacated but sits behind DeVonta Smith at DC2 on a 51%-
  pass offense, while Carnell Tate (pick 4) has only 83 vacated yet is already
  DC1 on a 60%-pass team. The raw number says Lemon; the situation says Tate.
- **Veterans get the same treatment** (`draft.player_context`): every board row
  shows the room — `moved` (with the prior team), `blocked_by` (best OTHER
  player at his position, by last year's points; empty = leads the room),
  `vacated_fp`, `depth_rank`, `pass_rate`, `new_coach`. It reads coherently
  because it's all one join: DJ Moore CHI→BUF shows up as Rome Odunze's 262
  vacated AND as the man now blocking Khalil Shakir. Also context only, never a
  model input.
- **Health is the asterisk on every season projection** (`draft.injury_context`,
  the hover "i" on each board row). A season total silently assumes 17 games; the
  flag says when that's a stretch — `weeks_out`, the body part and round of his
  last Out/Doubtful report, `ended_hurt`, and current roster `status`. Three
  things it gets right that a naive version wouldn't:
  - `ended_hurt` is measured against **the team's** last week (18 if it missed the
    playoffs, 22 if it reached the Super Bowl), not the player's own last report —
    against his own it's trivially true for everyone. Getting this wrong flagged
    418 of 768 players; correct, it's 160.
  - The report doubles as an **absence log**. "Not injury related — personal
    matter" is dropped outright, and `Illness` still counts as a missed game but
    never sets `ended_hurt` — a week-18 flu says nothing about Week 1.
  - `status` on the target-season roster (RES/PUP/RET) is the freshest signal we
    have in July: a live snapshot, not last December. It surfaces even for players
    with no injury history at all.
  Only skill-position rows join (96.8% on gsis_id); the 31% overall rate is just
  `injuries` covering linemen and defense that `weekly` never kept. Context only,
  like the rest — the injury report is a coach's strategic document as much as a
  medical one, so as a feature it would mostly fit team reporting habits.
- `draft_picks` uses **PFR team codes** (GNB/KAN/LVR/NWE/NOR/SFO/TAM/LAR); the
  rest of the lake uses nflverse codes. `_PFR_TEAM` maps them — without it, eight
  teams silently lose all team context.
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
- **Draft-and-win backtest** (`backtest_draft.py`, `python -m ffdata.backtest_draft
  --season 2024 --sims 200`): the honest end-to-end test of the stack. Drafts a
  team from `draft_board(season)` (preseason, leak-free) while the other managers
  draft off a **naive last-year-points** board, then replays the season's ACTUAL
  weekly results — setting each week's lineup with the same greedy fill the
  optimizer uses — through a round-robin schedule + single-elim playoffs to a
  champion. Randomizes the draft slot over `sims` runs → a title/playoff *rate*,
  not one lucky season. Each sim also runs a control where our slot drafts naively
  too, so the reported **lift** (title-rate, playoff-rate, mean-finish) isolates
  what our value model adds over the baseline on identical schedules. Leak-free by
  construction: the draft never sees the weekly points. The pure sim engine
  (`snake_order`/`run_snake_draft`/`best_week_total`/`replay`/`round_robin`/
  `standings`/`playoffs`/`simulate_season`) is unit-tested; **`run_backtest` needs
  the lake and is unvalidated here** (no data) — and its numbers are only as good
  as the projections feeding it. `prop_accuracy(season)` reports per-market
  projection MAE + P(over) calibration (reusing the prop engine); hit-rate-vs-book
  can't be computed (nflverse ships no odds), so calibration is the honest stand-
  in. Scope: K/DEF aren't drafted (streamed), so the sim uses the skill board.
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
