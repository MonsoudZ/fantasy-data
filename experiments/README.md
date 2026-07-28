# Experiment results

Generated from the JSON artifacts in `results/`. Regression holdouts may be rerun; locked holdouts are reserved for one final evaluation.

| Date | Experiment | Variant | Holdout | Code | Result |
|---|---|---|---|---|---|
| 2026-07-28 | strategy-sweep | 2022–2025 · 0–100% sharp · paired | 2025 (regression) | `b4628248` | 100% sharp adaptive: finish 5.3958 vs 6.5208 (Δ -1.12); playoffs 58.3%; titles 8.3% |
| 2026-07-28 | season-sweep | 2022–2025 · 0–100% sharp · baseline | 2025 (regression) | `4a34737f` | 0% sharp: finish 3.4167, playoffs 93.8%, titles 54.2%; 100%: finish 6.5208, playoffs 50.0%, titles 6.2% |
| 2026-07-28 | season | sharp · baseline | 2025 (regression) | `86bba07e` | finish 6.2 [4.5, 8.2]; playoffs 50.0% [25.0, 75.0]; titles 16.7% [0.0, 41.7] |
| 2026-07-28 | season | naive · baseline | 2025 (regression) | `07f2c6f4` | finish 2.3 [1.8, 3.0]; playoffs 100.0% [100.0, 100.0]; titles 66.7% [41.7, 91.7] |
| 2026-07-28 | weekly | LightGBM vs trailing | 2025 (regression) | `b9f0f1a9` | LightGBM MAE 4.467, RMSE 6.1, rank 0.6772 |
| 2026-07-28 | draft | realistic · ppr | 2025 (regression) | `abfd320f` | rank 0.7819 vs prior 0.766; MAE 39.62 |
