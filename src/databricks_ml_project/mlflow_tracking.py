"""Advanced MLflow tracking utilities.

This module focuses on:
- nested experiment tracking,
- custom metric logging,
- artifact logging,
- custom pyfunc model registration workflows.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd


class ThresholdPyFuncModel(mlflow.pyfunc.PythonModel):
    """Simple custom real-time model object.

    This class emulates a custom business-rule model and is intentionally simple
    so the packaging + registration pattern is easy to understand.
    """

    def load_context(self, context):
        with open(context.artifacts["threshold_config"], "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.threshold = cfg["threshold"]

    def predict(self, context, model_input: pd.DataFrame):
        return (model_input["score"] >= self.threshold).astype(int)


def log_experiment_artifacts(metrics: dict, params: dict, artifact_payload: dict) -> None:
    """Log custom run data in a reusable helper."""

    with mlflow.start_run(run_name="custom_tracking"):
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "experiment_notes.json"
            artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(artifact_path), artifact_path="notes")


def log_and_register_custom_pyfunc(model_name: str, threshold: float = 0.7) -> str:
    """Package and register a custom pyfunc model in Unity Catalog registry."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = Path(tmp_dir) / "threshold_config.json"
        cfg_path.write_text(json.dumps({"threshold": threshold}), encoding="utf-8")

        with mlflow.start_run(run_name="register_custom_pyfunc"):
            model_info = mlflow.pyfunc.log_model(
                artifact_path="threshold_model",
                python_model=ThresholdPyFuncModel(),
                artifacts={"threshold_config": str(cfg_path)},
                registered_model_name=model_name,
                input_example=pd.DataFrame({"score": [0.2, 0.8]}),
            )

    return model_info.model_uri
