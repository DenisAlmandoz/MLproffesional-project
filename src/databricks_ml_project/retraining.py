"""Automated retraining decision logic.

This module demonstrates how to trigger retraining based on:
- drift threshold breaches,
- degraded performance metrics,
- and then select the top model candidate from a training tournament.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class CandidateModel:
    model_uri: str
    metric_name: str
    metric_value: float


def should_trigger_retraining(drift_alert: bool, current_auc: float, auc_floor: float = 0.75) -> bool:
    """Decide whether to launch retraining workflow."""

    return drift_alert or (current_auc < auc_floor)


def select_best_candidate(candidates: Iterable[CandidateModel]) -> CandidateModel:
    """Select best model by maximizing metric value.

    In a real pipeline this can include tie-breaking on fairness, latency, cost,
    or slice-level performance.
    """

    candidates = list(candidates)
    if not candidates:
        raise ValueError("No candidate models provided")

    return max(candidates, key=lambda c: c.metric_value)
