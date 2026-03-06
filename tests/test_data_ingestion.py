import csv
from pathlib import Path

from databricks_ml_project.data_ingestion import generate_synthetic_events_csv, summarize_events


def test_generate_synthetic_events_csv_creates_expected_rows(tmp_path: Path):
    output = tmp_path / "events.csv"
    result_path = generate_synthetic_events_csv(str(output), n_rows=20, seed=1)

    assert Path(result_path).exists()

    with open(result_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 20
    assert {"customer_id", "event_ts", "amount", "region", "label"}.issubset(reader[0].keys())


def test_summarize_events_counts_records_and_labels():
    events = [{"label": 0}, {"label": 1}, {"label": 1}]
    summary = summarize_events(events)

    assert summary["count"] == 3
    assert summary["positive_labels"] == 2
