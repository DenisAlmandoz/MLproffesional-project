"""Deployment helpers for Databricks Model Serving.

This module provides rollout payload examples for:
- canary deployment (small percentage to new model),
- blue/green style switching,
- query helpers for REST and SDK patterns.
"""

from __future__ import annotations


def canary_traffic_config(current_model: str, challenger_model: str, challenger_pct: int = 10) -> dict:
    """Build endpoint traffic config with canary split."""

    return {
        "served_entities": [
            {"entity_name": current_model, "entity_version": "Production", "name": "stable"},
            {"entity_name": challenger_model, "entity_version": "Candidate", "name": "canary"},
        ],
        "traffic_config": {
            "routes": [
                {"served_model_name": "stable", "traffic_percentage": 100 - challenger_pct},
                {"served_model_name": "canary", "traffic_percentage": challenger_pct},
            ]
        },
    }


def blue_green_cutover_config(new_model: str) -> dict:
    """Build simplified cutover payload where all traffic goes to green deployment."""

    return {
        "served_entities": [{"entity_name": new_model, "entity_version": "Production", "name": "green"}],
        "traffic_config": {"routes": [{"served_model_name": "green", "traffic_percentage": 100}]},
    }


def rest_scoring_payload(instances: list[dict]) -> dict:
    """Create REST payload body for custom pyfunc scoring."""

    return {"dataframe_records": instances}
