"""Central configuration helpers for Databricks ML project.

This module keeps environment-specific values in one place to make promotion
across dev/test/prod safer and easier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    """Runtime config for data, MLflow, and deployment assets.

    Keeping these values immutable avoids accidental mutation during job runs.
    """

    catalog: str = os.getenv("DBX_CATALOG", "main")
    schema: str = os.getenv("DBX_SCHEMA", "ml_exam")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "/Shared/ml_exam_project")
    model_name: str = os.getenv("MODEL_NAME", "main.ml_exam.exam_candidate_model")

    @property
    def feature_table(self) -> str:
        return f"{self.catalog}.{self.schema}.customer_features"

    @property
    def training_table(self) -> str:
        return f"{self.catalog}.{self.schema}.training_events"

    @property
    def inference_table(self) -> str:
        return f"{self.catalog}.{self.schema}.inference_logs"


CONFIG = ProjectConfig()
