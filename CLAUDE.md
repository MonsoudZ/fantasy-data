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
`python -m ffdata.cli --datasets rosters --seasons 2026` (for a 2026 draft),
plus `python -m ffdata.cli --live` for today's IR/PUP/suspension feed.
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
| `backtest_draft.py` | draft-and-win backtest; grades with a HINDSIGHT lineup (isolates draft value only) |
| `season_sim.py` | blind season replay: all 12 teams drafted + managed on projections; naive/sharp fields |
| `betting.py` | American-odds / de-vig math + empirical P(over), shared by props/gamelines |
| `props.py` | player-prop edge finder (per-stat models; you supply odds) |
| `gamelines.py` | game total/spread/moneyline forecast vs market (informational; lines from `schedules`) |
| `draft.py` | preseason season projections, VOR, snake/auction, keepers, trades, rookies |
| `dynasty.py` | age curves (delta method) + multi-year dynasty value |
| `store.py` | JSON persistence for saved leagues + lineup teams (incl. custom scoring) |
| `sleeper.py` | import a league from Sleeper's public API; **live availability feed** (today's IR/PUP/suspensions) |
| `advice.py` | grounded Claude explanations of compare/keeper/trade decisions (opt-in) |
| `web.py` `static/index.html` | FastAPI + tabbed UI; every player entry is a search picker, never typed |

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
- **Never hardcode a season.** `MatchupSimulator.fit` defaulted `resid_seasons`
  to a literal `[2023, 2024]` and derived the feature range from it, so the frame
  froze at 2024 and *every* later season became unprojectable — `project()` got
  an empty test set and LightGBM raised "Input data must be 2 dimensional and non
  empty", which killed the **whole lineup optimizer and props tab** the moment
  2024 ended. Now `matchup.fit_seasons()` derives it from `current_nfl_season()`
  (pinned by a test). `gamelines.py` had the right pattern all along: take the
  seasons from the data (`sorted(train["season"].unique())[-2:]`).

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
- **The draft edge is almost entirely "don't hoard QBs."** `season_sim` plays a
  real past season blind: draft off the preseason board, then manage every one of
  the 12 teams on projections only (start-by-projection, worst-first waivers on
  FORM not one noisy week, actual points read solely to grade a locked lineup).
  Measured over **48 runs per field** — 2022/23/24/25 × all 12 draft slots,
  1QB/2RB/2WR/1TE/FLEX/DEF/K + 5 bench, standard scoring — against two opponent
  models (`opponent=`):

  | vs the field | mean finish | playoffs | titles | (chance) |
  |---|---|---|---|---|
  | **naive** (rank by raw prior points) | 6.15 | 56% | **4%** | 6.5 / 50% / 8.3% |
  | **sharp** (draft our VOR board + per-team noise) | 6.00 | 56% | **10%** | " |

  **With realistic roster management, the board has no measurable edge** — mean
  finish ~6 of 12 (chance 6.5) and a title rate at the base rate, against *either*
  field. This is the honest number and it is much lower than earlier notes here.
  Two corrections got it there:
  1. **Only our team used to take waivers** — an unfair edge. Now all 12 manage.
  2. **The sim used to bench our drafted rookies** (a rookie with no games is
     absent from the trailing weekly model → projected 0 → sat), which flattered
     us: it quietly fielded replacement veterans instead of the busts we drafted.
     Now rookies start on their preseason prior (`_seed_rookie_prior`), as a real
     manager would — and they bust, which is where the edge went.
  The draft view (`format_league_report`) shows the cause directly: the VOR board,
  fed the **draft-capital rookie projections**, drafts rookies with *negative* VOR
  (2025 slot 1: Tetairoa McMillan −5, Emeka Egbuka −8, Cam Ward −69) because those
  projections are full-season and rookies underperform them early. **Open question
  worth a backtest: the rookie season-projection looks too high for redraft; a
  discount, or benching unproven rookies behind veterans, would likely recover
  most of the lost finish.** Still season-dependent (naive 2022 mean 3.75 / 8
  titles, 2023 mean 7.25 / 0) — one season is an anecdote, hence the 12-slot × 4-yr
  sweep.
  Two mechanisms the sweep exposed and now guards:
  - **Bye-week stud circulation.** Waivers on a single week's projection drop a
    stud who's on bye (projects ~0) for a streamer; studs then circulate the
    league on their byes and the title becomes a lottery. Fixed: waivers value a
    player by season-to-date FORM (`WAIVER_MIN_GAIN`, form-smoothing), so a bye
    barely moves him. Without it, moves ran ~16/team/season; with it, ~0-4.
  - **Draft ≠ the whole game.** Our 2024 1.01 was Christian McCaffrey (4 games
    played); the waiver rule dropped him in week 3 when his form collapsed. The
    naive field's QB-hoarding leaves us a bad QB (2025: Cam Ward), fixed on waivers
    by week 2 (Dak Prescott).
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
  In the offseason they differ. `current` is a **backend** concept — how far the
  played-data lake reaches — and must never surface in the UI.
- **One season, everywhere user-facing.** Every UI field, every API default and
  every user-facing CLI uses `upcoming_nfl_season()`. There is no season picker
  and no second season on screen: earlier seasons are training data the models
  read, never something the user selects. Showing last year's number beside this
  year's advice is exactly how you end up drafting for a season that already
  happened. `/api/config` returns a single `season` (plus `started`).
  What that means before kickoff, measured for 2026 in July:

  | source | 2026 | so |
  |---|---|---|
  | `rosters` / `depth_charts` | 2,930 / 3,100 rows | draft, keepers, trades, dynasty, rookies **live** |
  | `schedules` | 272 games, 67 with Vegas lines | game lines **live** |
  | `weekly` / `injuries` / `snap_counts` | **0 rows** | lineup + props **dormant** |

  Weekly stats only exist for seasons that have been PLAYED, so the two in-season
  tabs are disabled with a plain explanation rather than failing on an empty
  frame — and emphatically rather than serving last season's numbers under this
  season's label. `ingest.season_not_started()` is the single predicate; the web
  returns `{ok: false, not_started: true}` and the weekly CLIs exit with the same
  sentence. The ingest CLI still pulls `FIRST_SEASON..current` — that's the
  backend lake, and it's the one place `current` belongs. **The predicate is
  games-based, not calendar-based:** given a lake `con` it checks whether `weekly`
  actually has the season's rows, which is correct in the ~week between the Sept-1
  label rollover and real Week-1 kickoff (the month rule called that "started" and
  the weekly path then crashed on an empty frame); it falls back to the calendar
  only when there's no lake to consult (a pre-ingest CLI message).
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
- **Multi-year career + durability: better projection, but NOT more titles**
  (`draft._career_features`, `career=` flag, default OFF). Recency-weighted career
  form + games-played durability are leak-free (row S sees only seasons ≤ S) and
  genuinely improve the season projection every year — rank **0.734 → 0.749**, MAE
  **34.2 → 33.5** (standard, 2022-25 out of sample). Unlike the context features,
  this is a real accuracy gain. **But it FAILED the sim test**: 48 runs/field, it
  won *fewer* leagues — naive titles 33% → 17%, sharp 10% → 4%. The football
  reason is coherent: career/durability rewards proven, safe production and
  penalises the unproven, which raises accuracy but suppresses the **ceiling** —
  and a title usually needs a breakout that *exceeds* career norms. Floor beats
  bust but loses to upside. So `career` stays OFF everywhere until it earns its
  keep on the objective that matters; the feature is kept because the projection
  gain is real (useful if we ever rank on accuracy rather than title EV).
- **Upside RESCUES career, but the stack still ties the simple baseline**
  (`draft._upside_features`, `upside_weight=`, default 0). `_upside_features`
  measures the SHAPE of a player's weeks (leak-free, recency-weighted): `u_ceiling`
  = mean of his top-3 weekly scores (boom level), `u_floor` = bottom-3 (floor),
  `u_boom`, `u_stdev`. `draft_board(upside_weight=w)` adds a bonus lifting players
  whose weekly ceiling beats their position median, ON TOP of the accurate career
  mean — additive, so a high-mean player who also booms gets the biggest lift.
  Adding it undid career's floor damage (sharp titles 4% → 8%, naive 17% → 35%),
  confirming the floor+ceiling direction. But career+upside (0.10) vs the plain
  baseline over 48 runs/field: naive titles 35% vs 33%, **sharp 8% vs 10%** (one
  title in 48 — a tie), mean finish slightly worse both. A single-slot 2025 scan
  looked great (place 1.50) but was noise. **Net: neither career nor career+upside
  beats the simple projection in the sim** — the real sim winners are STRUCTURAL
  (roster-aware draft + rookie discount), not projection features, and the sim's
  variance floor sits above any projection edge this size. Both stay OFF by
  default. Their real value is DECISION SUPPORT for a human: career sharpens
  accuracy, upside surfaces sleepers the mean buries (2024 CMC reads a 21.2
  ceiling off 4 games) — a mechanical auto-manager can't exploit a sleeper the way
  a drafter does, so the right home is a board VIEW (accuracy + a ceiling/sleeper
  column), not an auto-applied weight.
- **Schedule-adjusting PRIOR production does NOT help projection**
  (`draft._schedule_features`, `_SCHEDULE_FEATS`, `schedule=` flag, default OFF).
  Neutralises a player's season-S points by the strength of the defenses he
  actually faced that year (leak-free: season-S defensive fp-allowed → per-position
  difficulty multiplier → `sched_faced` = mean difficulty faced, `p_fp_adj` =
  points divided by it). Also swaps the 0.6 blend anchor to `p_fp_adj` when on.
  Backtest 2022-25 out-of-sample, standard, players ≥20 fp — rank corr / MAE:
  baseline **0.650 / 42.9**, +schedule **0.648 / 42.8** (rank slightly *worse*),
  +career 0.666 / 42.2, +career+schedule 0.665 / 42.0 (identical to career; MAE
  −0.2 is noise). Why it fails: at the SEASON level `sched_faced` regresses to
  ~1.0 for every full-time starter (Drake Maye's "easiest schedule ever" 2025
  reads **1.04**), and the only extreme values are small-sample backups
  (Stidham 1.43 off ~1 game). Last year's schedule difficulty doesn't carry to
  next year — a player faces a *different* schedule. Code kept behind the default-
  OFF flag; **revisit only if fed NEXT year's actual 2026 schedule** (the
  strength-of-schedule the drafter cares about), not last year's, which needs the
  2026 opponent-strength estimate we don't have preseason.
- **Forward strength-of-schedule doesn't move rankings either — but it's now
  shown as CONTEXT** (`draft._def_difficulty` → `_forward_sos` →
  `schedule_context`; board badge 🟢/🔴). The honest forward version: estimate
  every defense's strength going INTO the draft year from its recency-weighted
  PRIOR seasons (normalized so 1.0 = league-average, leak-free, and it emits a
  row for the not-yet-played season so 2026 has a number), then average the
  difficulty of the opponents each team is actually scheduled to face. A crude
  1-year `sos` was already a model feature; swapping in this multi-year
  normalized version, OR applying it as a post-hoc projection multiplier at
  k=0.15/0.3/0.5, changed out-of-sample accuracy by nothing (rank 0.6503 →
  0.6504-0.6507, MAE flat, standard, 2022-25). **Why:** the entire 2026 spread
  is std ~0.02 — easiest road PHI **1.037**, toughest LV **0.963**, i.e. ±~5% at
  the extremes — because a full season of matchups nets out and next-year defense
  strength regresses hard. Schedule is a WEEKLY start/sit signal, not a
  season-ranking one. NB the "Drake Maye easiest schedule ever" narrative is
  about 2025 (already played); forward-looking, NE's 2026 QB road reads **0.976**
  (slightly *tougher* than average). So it's surfaced like `blocked_by` /
  injuries — a per-player `sched_ahead` / `sched_rank` **tiebreaker between
  similar players, never folded into VOR** (the badge only lights at the top/
  bottom of a position, and its tooltip says so outright).
- **SOS done the Sharp way DOES move ranking — it was the METHOD, not the concept**
  (`draft._sos_quality`, feature `sos_q`, shipped IN `_FEATS`). Prompted by
  Sharp Football's 2026 SOS (they use Vegas win totals and note prior-year
  *records* explain only 3.9% of actual SOS). The position-specific fp-allowed
  SOS above is noise; rating opponents by **overall team quality** —
  recency-weighted prior **point-differential per game** — is a far more stable
  estimate and clears the bar: on the career+comp board stack, rank
  **0.6664 → 0.6677** (+0.0013, ~2× the competition gain), positive in 3 of 4
  seasons (2022 +0.0031, 2024 +0.0053, 2025 −0.0042), MAE flat. It's
  COMPLEMENTARY to the fp-based `sos` (add 0.6677 > replace 0.6671), so both are
  kept. Leak-free (strictly-prior seasons, recency-weighted) and it emits a 2026
  row. Its 2026 easiest list — CLE, NO, PHI, DET, BAL — now AGREES with Sharp
  (DET, NO, CIN, CLE, NYJ), where the fp version buried DET at #22. Lesson: a
  weak SOS measured dead ≠ SOS is dead; the opponent-strength ESTIMATE is what
  matters. Ceiling is still modest (schedule averages out over 17 games) — when a
  season win-totals / full game-lines source lands, swap point-diff for market
  strength and re-measure.
- **Competition / opportunity: a small but REAL projection gain, and the best
  context signal yet** (`draft._competition_features`, `_COMPETITION_FEATS`,
  `competition=` flag; `competition_context` + 📈 board badge). Fantasy points
  follow VOLUME, and volume moves when the players competing for touches leave or
  arrive — something a player's own prior stats can't see. Leak-free (prior-year
  volume + the preseason-known target-year roster; keyed by TARGET season):
  `opp_share` = his share of the volume returning to his position room,
  `vac_share` = fraction of his team's prior position volume that vacated,
  `comp_vol` = raw prior volume of the others in the room now. Backtest 2022-25
  out of sample, standard, ≥20 fp — rank corr / MAE:
  baseline **0.6503 / 42.89**, +comp **0.6514 / 42.70**, +career 0.6657 / 42.20,
  **+career+comp 0.6664 / 42.06**. Unlike schedule it moves BOTH metrics the
  right way, and it helps RB specifically (RB rank 0.5834 → **0.5861** on the
  career stack — where touch competition actually lives). NB comp-ALONE *hurts*
  RB rank (0.5791 → 0.5737); it only helps stacked on career, so it's enabled on
  the board (always `career=True`) but NOT added to the sim's non-career board.
  The gain is small (+0.0007 rho on career) so the honest headline is "marginal
  for projection", but it's the first such signal that's net-positive everywhere
  it's on. Its bigger value is CONTEXT: it quantifies exactly what a drafter says
  out loud — Gibbs 2026 reads `opp_share` 0.54→**0.70**, `vac_share` **0.36**
  (Montgomery's touches gone), competition 268→147; Bijan 0.71→**0.76** / 0.28.
  The 📈 badge lights when ≥20% of the room vacated (`_OPP_OPEN_SHARE`, ~15 of the
  top 60), and unlike schedule its tooltip says the projection ALREADY reflects
  it. Board runs `draft_board(..., career=True, competition=True)`.
- **Scheme / pace (team pass-rate + plays-per-game) does NOT earn a spot in the
  projection — but it's already the right kind of context.** Tested the ENTERING
  team's prior-year pass share and pace as features (leak-free: target-team's
  S-1 `t_pass_rate` / `t_plays_pg` → predict S). Backtest 2022-25, standard,
  ≥20 fp: **alone it HURTS** (rho 0.6503 → 0.6493, RB 0.5791 → 0.5749), and on
  the career+comp board stack it's a +0.0006 whisper (0.6664 → 0.6670). The
  reason is double-counting: a player's own prior targets/carries already encode
  his scheme, so the team's raw rate is mostly redundant noise. The one place it
  IS real is **team-changers** (new environment his own history can't show):
  splitting the board stack by `team_changed`, +scheme moves MOVERS +0.0064
  (0.5179 → 0.5243) but STAYERS −0.0012 (0.7030 → 0.7018) — a genuine signal on
  ~25% of players, drowned by the majority it double-counts. Net below the bar
  (competition cleared it by helping RB broadly and never hurting a subset;
  scheme degrades stayers), so it stays OUT of the projection. It's already
  surfaced where it belongs: `player_context.pass_rate` (the "54% pass" on every
  board row) reads the player's NEW team's prior pass rate — exactly the number a
  drafter wants for a receiver who changed teams. No code added; context suffices.
- **Workload / injury-history durability does NOT improve ranking** (prototype
  only, not shipped). Tested the "curse of 370" intuition — a huge touch count or
  a fragile track record predicting a down/absent year (the CMC case: 429 touches
  in 2019 → 3 games in 2020; 440 in 2025). Features: this-year touches, career
  recency-weighted touches, games missed this year + career, peak-ever workload.
  On the career+comp board stack, 2022-25 standard: rank **0.6664 → 0.6664**
  (dead flat) and RB **0.5861 → 0.5833** (worse). Why: `career`'s `c_games_avg` /
  `c_games_min` and the raw `p_games` already encode durability, so an explicit
  injury-proneness feature is redundant; and a healthy-but-fragile profile (CMC
  2026) does NOT reliably predict a down year across all backs — injury is mostly
  random year-to-year, the anecdote doesn't generalize. Baking it into VOR would
  make projections WORSE, so it stays out. Current-status availability (IR/PUP/
  suspended) still lowers VOR via `availability_penalty`; historical proneness is
  best left as the injury CONTEXT already on each row, for the human to judge.
- **Veterans get the same treatment** (`draft.player_context`): every board row
  shows the room — `moved` (with the prior team), `blocked_by` (best OTHER
  player at his position, by last year's points; empty = leads the room),
  `vacated_fp`, `depth_rank`, `pass_rate`, `new_coach`. It reads coherently
  because it's all one join: DJ Moore CHI→BUF shows up as Rome Odunze's 262
  vacated AND as the man now blocking Khalil Shakir. Also context only, never a
  model input.
- **Health is the asterisk on every season projection** (`draft.availability_context`,
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
  `rosters` is **weekly** (a player goes ACT→DEV→INA within a season), so status
  must come from his LAST known week — `any_value()` reports a status he left.
- **Suspensions live in Sleeper, not nflverse** (`sleeper.refresh_live_status` →
  the `sleeper_status` view → `draft._live_status`). nflverse's `rosters.status`
  has the right codes (`SUS`, `RSN` = did not report, `NWT` = not with team) and
  was populated densely in 2019–20 (187/228/177 players in 2019) — then stopped:
  **one** SUS row in 2022, zero in 2021 and 2023–26. They're still mapped in
  `_INACTIVE_STATUS` (correct where data exists) but will never fire on a current
  draft. Sleeper's public API fills the gap and is the only source here that knows
  about **today**:
  - Suspensions are under **`injury_status = "Sus"`**, *not* the top-level
    `status` field, which only ever reads Active/Inactive. Also `DNR` (did not
    report), `IR`, `PUP`, `NA`, `COV`, `Out`, `Questionable`.
  - **Do not join on `gsis_id`** — Sleeper populates it for only ~16% of rostered
    players and some carry stray whitespace. Join on name+position: 88% with zero
    collisions. Both sides must go through `sleeper.norm_name` (it strips Jr./Sr./
    numeral suffixes) or the keys drift.
  - Sleeper ships literal **"Duplicate Player"** placeholder rows — drop them.
  - `news_updated` dates each record (73 of 76 flags were current when added), so
    a stale flag is distinguishable from a live one. `injury_start_date` is always
    empty — don't rely on it.
  - Kept for **every position**, not just skill: a suspended tackle feeds
    `line_context`. Refresh with `python -m ffdata.cli --live` (12h TTL; Sleeper
    asks for ≤1 call/day). The board reads the cached view and never fetches, so
    it stays fast and works offline. The web's context cache keys on the
    `sleeper_status.parquet` **mtime**, so a refresh is picked up on the next
    request — a plain per-season cache would freeze the "live" feed until the
    server restarted.
  - Complementary, not a replacement: nflverse tells you how last season *ended*,
    Sleeper what's true *now*. De'Von Achane reads "ruled out wk 18, shoulder" from
    one and "questionable — Shoulder — Surgery, reported 2026-07-19" from the other.
  - Honest caveat: right now that's **2 suspended players, both defensive**, so
    the suspension flag shows nothing on a fantasy board today. The live *injury*
    feed is where the value is (Mahomes: Knee-ACL/Surgery; 20 PUP, 18 IR).
- **The offensive line matters, but only past a threshold** (`draft.line_context`).
  Linemen never appear in `weekly` (ingest keeps skill positions), but
  `depth_charts` + `injuries` carry every position, so the unit is recoverable.
  Measured over 3,182 team-weeks 2019–24, each team compared to its **own** season
  average so team quality cancels:

  | starting OL ruled Out | 0 | 1 | 2 | 3 |
  |---|---|---|---|---|
  | team RB pts vs usual | +0.03 | +0.33 | **−3.72** | −4.65 |

  One lineman down is *nothing*; two costs a backfield ~3.8 PPR pts/game
  (t = −3.84, 95% CI [−5.8, −1.9]), and it replicates in both halves of the era
  (−3.3 in 2019–21, −4.4 in 2022–24). A plain correlation reads **−0.03** and
  would have thrown it away — the relationship is a threshold, not a gradient.
  That threshold is **enforced, not just documented**: `line_context` drops any
  team below `_OL_THRESHOLD` (2) starters out, so the board never flags a single
  injury the data says is noise.
  Rides on **RB rows only**: QBs showed nothing (−0.40 at two down). Preseason
  caveat: in July it's driven by linemen who ended last season hurt (2026: 11
  teams have one, only NYG has two), so it earns its keep in-season.
- **Two unit-level things measured as nothing and are deliberately not shipped:**
  - *OL continuity* (how many of the five starters return): r = **−0.06** vs RB
    point change over 192 team-seasons, non-monotone, and the sign is backwards —
    it's regression to the mean, not blocking.
  - *Opposing defenders out*: the gradient looks right (−0.48 → +0.91 → +3.46 as
    2 starters sit) and 2+ gives +3.77 pts (t = 2.10), but it **flips sign across
    halves of the era** (−1.2 in 2019–21, +6.8 in 2022–24). Not a finding.
  Depth charts changed format mid-stream: 2019–24 are weekly rows on
  `depth_position`/`depth_team`/`club_code`, 2025+ are dated snapshots on
  `pos_abb`/`pos_rank`/`team`. Any multi-season depth query must read both. The
  snapshot files stack many dates; `_normalize_depth_charts` keeps the latest
  **per team** (not one global `max(dt)`, which would drop every team whose chart
  was refreshed on an earlier date). `ended_hurt` takes a team's final week from
  `schedules` (how far it actually went), not from the last injury-report week;
  and `new_coach` anchors on the coach a team **ended** the season with (last
  game, week-desc), so a mid-season firing labels the change correctly.
- **No player is ever typed.** Every spot that used to take a name — keepers,
  both trade sides, compare, waiver exclusions, prop lines — is a search picker
  over the list we already have (`picker()` in `index.html`, one component, two
  sources: `/api/players` for weekly and `/api/names` for the season-long board).
  A misspelt name used to silently vanish from a keeper set or never price a prop.
  `/api/names` returns the **whole** board, not the top-N the UI displays, or a
  keeper outside the top 50 couldn't be selected; it reuses the cached board so
  it's only slow once. The props builder narrows each player's market list via
  `/api/markets` (no QB receptions) and still serialises to the same CSV the
  server parses — "paste CSV" toggles the raw box for bulk entry.
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
