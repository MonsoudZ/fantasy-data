# Experiment results

Generated from the JSON artifacts in `results/`. Regression holdouts may be rerun; locked holdouts are reserved for one final evaluation.
Superseded results are retained for auditability but must not be used for conclusions.

| Date | Experiment | Variant | Status | Holdout | Code | Result |
|---|---|---|---|---|---|---|
| 2026-07-29 | strategy-sweep | 2022–2025 · 0–100% sharp · paired | Recorded | 2025 (regression) | `92cfa6c1` | 100% sharp adaptive: finish 5.4375 vs 4.375 (Δ +1.06); playoffs 66.7%; titles 16.7% |
| 2026-07-29 | season-sweep | 2022–2025 · 0–100% sharp · baseline | Recorded | 2025 (regression) | `dabbbf7c` | 0% sharp: finish 3.4167, playoffs 93.8%, titles 54.2%; 100%: finish 4.375, playoffs 81.2%, titles 20.8% |
| 2026-07-28 | strategy-sweep | 2022–2025 · 0–100% sharp · paired | **Superseded** | 2025 (regression) | `b4628248` | The strategy comparison used the flawed sharp-manager selection model. [Corrected result](results/20260729T172835Z-strategy-sweep-2025-94af2eb5.json). |
| 2026-07-28 | season-sweep | 2022–2025 · 0–100% sharp · baseline | **Superseded** | 2025 (regression) | `4a34737f` | Sharp-manager noise reordered the board but did not affect actual roster-aware selections. [Corrected result](results/20260729T171613Z-season-sweep-2025-a81fc4da.json). |
| 2026-07-28 | season | sharp · baseline | **Superseded** | 2025 (regression) | `86bba07e` | Sharp-manager noise reordered the board but did not affect actual roster-aware selections. [Corrected result](results/20260729T171613Z-season-sweep-2025-a81fc4da.json). |
| 2026-07-28 | season | naive · baseline | Recorded | 2025 (regression) | `07f2c6f4` | finish 2.3 [1.8, 3.0]; playoffs 100.0% [100.0, 100.0]; titles 66.7% [41.7, 91.7] |
| 2026-07-28 | weekly | LightGBM vs trailing | Recorded | 2025 (regression) | `b9f0f1a9` | LightGBM MAE 4.467, RMSE 6.1, rank 0.6772 |
| 2026-07-28 | draft | realistic · ppr | Recorded | 2025 (regression) | `abfd320f` | rank 0.7819 vs prior 0.766; MAE 39.62 |
