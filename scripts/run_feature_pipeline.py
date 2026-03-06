"""Databricks Job entry point for feature engineering.

Run this as a task in Databricks Workflows. The script intentionally documents
all steps for exam-aligned study and operational clarity.
"""

from databricks_ml_project.config import CONFIG
from databricks_ml_project.feature_pipeline import prepare_online_table_note


if __name__ == "__main__":
    print(f"Feature table target: {CONFIG.feature_table}")
    print(prepare_online_table_note())
