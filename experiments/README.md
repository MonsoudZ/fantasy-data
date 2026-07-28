# Experiment results

Generated from the JSON artifacts in `results/`. Regression holdouts may be rerun; locked holdouts are reserved for one final evaluation.

| Date | Experiment | Holdout | Code | Result |
|---|---|---|---|---|
| 2026-07-28 | season | 2025 (regression) | `07f2c6f4` | finish 2.3 [1.8, 3.0]; playoffs 100.0% [100.0, 100.0]; titles 66.7% [41.7, 91.7] |
| 2026-07-28 | weekly | 2025 (regression) | `b9f0f1a9` | LightGBM MAE 4.467, RMSE 6.1, rank 0.6772 |
| 2026-07-28 | draft | 2025 (regression) | `abfd320f` | rank 0.7819 vs prior 0.766; MAE 39.62 |
