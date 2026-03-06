"""Distributed hyperparameter tuning with Optuna + MLflow nested runs.

This module demonstrates a pragmatic pattern:
- Parent MLflow run tracks the overall optimization campaign.
- Nested MLflow runs track each trial.
- Objective function can call Spark or single-node training logic.
"""

from __future__ import annotations

import mlflow
import optuna


def run_optuna_study(objective_fn, n_trials: int = 20):
    """Run Optuna study and log nested runs for each trial."""

    with mlflow.start_run(run_name="optuna_parent"):
        mlflow.log_param("optimizer", "optuna")
        mlflow.log_param("n_trials", n_trials)

        def wrapped_objective(trial: optuna.Trial) -> float:
            with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
                score = objective_fn(trial)
                mlflow.log_metric("objective_score", score)
                for key, value in trial.params.items():
                    mlflow.log_param(key, value)
                return score

        study = optuna.create_study(direction="maximize")
        study.optimize(wrapped_objective, n_trials=n_trials)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_value", study.best_value)

    return study
