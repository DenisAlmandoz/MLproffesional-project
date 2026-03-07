from databricks_ml_project.deployment import canary_traffic_config


def test_canary_traffic_percentages_sum_to_100():
    config = canary_traffic_config("main.model", "main.model_candidate", challenger_pct=15)
    traffic_sum = sum(route["traffic_percentage"] for route in config["traffic_config"]["routes"])
    assert traffic_sum == 100
