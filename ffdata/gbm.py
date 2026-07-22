"""Shared LightGBM defaults.

Every GBM in the codebase (weekly projections, season/draft, rookies, props)
wants the same base config -- learning rate, subsample, column sampling, a fixed
seed, quiet logging -- and differs only in a few knobs. Keeping the common part
here stops those dicts from drifting apart; callers pass the model-specific
overrides (n_estimators, num_leaves, min_child_samples, ...).
"""

from __future__ import annotations


def gbm_params(**overrides) -> dict:
    """Common LGBMRegressor params merged with per-model `overrides`."""
    params = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                  random_state=0, n_jobs=-1, verbose=-1)
    params.update(overrides)
    return params
