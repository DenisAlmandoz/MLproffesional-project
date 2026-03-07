"""Databricks Job entry point for model training and tracking."""

from databricks_ml_project.config import CONFIG


if __name__ == "__main__":
    print(f"Training table: {CONFIG.training_table}")
    print(f"Experiment name: {CONFIG.experiment_name}")
