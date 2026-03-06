"""Databricks Job entry point for monitoring workflow."""

from databricks_ml_project.config import CONFIG
from databricks_ml_project.monitoring import build_monitor_payload


if __name__ == "__main__":
    payload = build_monitor_payload(CONFIG.inference_table)
    print("Create/refresh Lakehouse monitor with payload:")
    print(payload)
