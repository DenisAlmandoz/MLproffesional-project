"""Distributed hyperparameter tuning with Ray Tune.

Ray is useful when you need flexible distributed training orchestration for
non-Spark-native training loops (e.g., PyTorch, XGBoost custom loops).
"""

from __future__ import annotations

from ray import tune
from ray.tune.search.optuna import OptunaSearch


def run_ray_tuning(trainable, num_samples: int = 8):
    """Execute Ray Tune search with Optuna backend search strategy."""

    search = OptunaSearch(metric="score", mode="max")

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(search_alg=search, num_samples=num_samples),
        param_space={
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "max_depth": tune.choice([3, 5, 8]),
            "min_child_weight": tune.choice([1, 3, 5]),
        },
    )

    results = tuner.fit()
    return results.get_best_result(metric="score", mode="max")
