"""Databricks Job entry point for retraining decision workflow."""

from databricks_ml_project.retraining import should_trigger_retraining


if __name__ == "__main__":
    trigger = should_trigger_retraining(drift_alert=True, current_auc=0.79)
    print(f"Retraining trigger decision: {trigger}")
