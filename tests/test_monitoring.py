from databricks_ml_project.monitoring import (
    evaluate_alert,
    numerical_drift_ks_statistic,
    population_stability_index,
)


def test_population_stability_index_non_negative():
    baseline = [1, 2, 3, 4, 5]
    current = [1, 2, 2, 4, 6]
    psi = population_stability_index(baseline, current)
    assert psi >= 0


def test_ks_statistic_range():
    ks = numerical_drift_ks_statistic([1, 1, 1], [2, 2, 2])
    assert 0 <= ks <= 1


def test_evaluate_alert():
    result = evaluate_alert("psi", value=0.3, threshold=0.2)
    assert result.is_alert
