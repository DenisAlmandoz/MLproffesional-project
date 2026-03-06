from databricks_ml_project.retraining import CandidateModel, select_best_candidate, should_trigger_retraining


def test_should_trigger_retraining_on_drift():
    assert should_trigger_retraining(drift_alert=True, current_auc=0.90)


def test_select_best_candidate():
    candidates = [
        CandidateModel(model_uri="models:/m1/1", metric_name="auc", metric_value=0.81),
        CandidateModel(model_uri="models:/m2/1", metric_name="auc", metric_value=0.85),
    ]
    best = select_best_candidate(candidates)
    assert best.model_uri == "models:/m2/1"
