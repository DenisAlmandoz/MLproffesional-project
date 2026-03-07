"""Lakehouse monitoring + drift metrics utilities.

Includes:
- simple numerical drift checks (KS test, with safe fallback),
- PSI drift checks,
- monitor configuration payload scaffolds,
- alert threshold evaluation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class DriftResult:
    metric_name: str
    value: float
    threshold: float

    @property
    def is_alert(self) -> bool:
        return self.value > self.threshold


def _to_sorted_float_list(values: Iterable[float]) -> list[float]:
    """Convert values into sorted float list for deterministic drift math."""

    return sorted(float(v) for v in values)


def numerical_drift_ks_statistic(baseline: Iterable[float], current: Iterable[float]) -> float:
    """Compute a Kolmogorov-Smirnov style statistic without external dependencies.

    The result is the maximum difference between empirical CDFs.
    """

    baseline_sorted = _to_sorted_float_list(baseline)
    current_sorted = _to_sorted_float_list(current)

    if not baseline_sorted or not current_sorted:
        raise ValueError("baseline and current must both contain values")

    all_values = sorted(set(baseline_sorted + current_sorted))
    n_base, n_curr = len(baseline_sorted), len(current_sorted)

    max_diff = 0.0
    for value in all_values:
        base_cdf = sum(v <= value for v in baseline_sorted) / n_base
        curr_cdf = sum(v <= value for v in current_sorted) / n_curr
        max_diff = max(max_diff, abs(base_cdf - curr_cdf))

    return max_diff


def population_stability_index(baseline: Iterable[float], current: Iterable[float], bins: int = 10) -> float:
    """Compute PSI for numeric or encoded categorical values without numpy."""

    baseline_values = [float(v) for v in baseline]
    current_values = [float(v) for v in current]

    if not baseline_values or not current_values:
        raise ValueError("baseline and current must both contain values")

    min_v, max_v = min(baseline_values), max(baseline_values)
    if min_v == max_v:
        return 0.0

    width = (max_v - min_v) / bins
    edges = [min_v + i * width for i in range(bins + 1)]

    def hist(values: list[float]) -> list[int]:
        counts = [0] * bins
        for value in values:
            idx = min(int((value - min_v) / width), bins - 1)
            counts[idx] += 1
        return counts

    base_hist = hist(baseline_values)
    curr_hist = hist(current_values)

    base_total = sum(base_hist)
    curr_total = sum(curr_hist)

    psi = 0.0
    for b, c in zip(base_hist, curr_hist):
        base_pct = max(b / base_total, 1e-8)
        curr_pct = max(c / curr_total, 1e-8)
        psi += (curr_pct - base_pct) * __import__("math").log(curr_pct / base_pct)

    return float(psi)


def build_monitor_payload(table_name: str, assets_dir: str = "/Workspace/Shared/monitoring") -> dict:
    """Create a workspace API payload template for Lakehouse Monitoring."""

    return {
        "table_name": table_name,
        "assets_dir": assets_dir,
        "output_schema_name": "monitoring",
        "inference_log": {"problem_type": "classification", "prediction_col": "prediction"},
        "slicing_exprs": ["region", "country"],
    }


def evaluate_alert(metric_name: str, value: float, threshold: float) -> DriftResult:
    """Wrap metric threshold comparisons for downstream notifications."""

    return DriftResult(metric_name=metric_name, value=value, threshold=threshold)
