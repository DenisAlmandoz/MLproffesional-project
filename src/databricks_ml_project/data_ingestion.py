"""Data generation and ingestion utilities for Databricks workflows.

Why this module exists:
- The project now includes built-in sample data generation so you can run end-to-end
  flows without depending on an external source system.
- It demonstrates a common lakehouse ingestion pattern: raw events -> bronze table.

The functions are intentionally verbose and heavily commented for learning purposes.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def generate_synthetic_events_csv(output_path: str, n_rows: int = 1000, seed: int = 42) -> str:
    """Generate a reproducible synthetic event dataset in CSV format.

    The generated dataset includes fields needed by the rest of the project:
    - customer_id
    - event_ts
    - amount
    - region
    - label

    Args:
        output_path: Target CSV path.
        n_rows: Number of records to emit.
        seed: Random seed for deterministic generation.

    Returns:
        The file path written, as a string.
    """

    random.seed(seed)
    regions = ["NA", "EMEA", "APAC", "LATAM"]
    start_time = datetime(2025, 1, 1, 0, 0, 0)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["customer_id", "event_ts", "amount", "region", "label"],
        )
        writer.writeheader()

        for i in range(n_rows):
            customer_id = f"C{random.randint(1, 250):04d}"
            event_ts = start_time + timedelta(minutes=15 * i)
            amount = round(max(random.gauss(120, 50), 1.0), 2)
            region = random.choice(regions)
            # This is a simple synthetic signal: larger amounts correlate with label=1.
            label = 1 if amount > 135 else 0

            writer.writerow(
                {
                    "customer_id": customer_id,
                    "event_ts": event_ts.isoformat(),
                    "amount": amount,
                    "region": region,
                    "label": label,
                }
            )

    return str(path)


def read_raw_events_csv(spark: "SparkSession", csv_path: str) -> "DataFrame":
    """Read generated raw CSV into a typed Spark DataFrame."""

    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    schema = T.StructType(
        [
            T.StructField("customer_id", T.StringType(), nullable=False),
            T.StructField("event_ts", T.TimestampType(), nullable=False),
            T.StructField("amount", T.DoubleType(), nullable=False),
            T.StructField("region", T.StringType(), nullable=True),
            T.StructField("label", T.IntegerType(), nullable=False),
        ]
    )

    return (
        spark.read.format("csv")
        .option("header", True)
        .schema(schema)
        .load(csv_path)
        .withColumn("ingest_ts", F.current_timestamp())
    )


def ingest_to_bronze_table(events_df: "DataFrame", bronze_table_name: str) -> None:
    """Write raw events DataFrame into a Delta bronze table."""

    (
        events_df.write.mode("append")
        .format("delta")
        .option("mergeSchema", "true")
        .saveAsTable(bronze_table_name)
    )


def ensure_training_table_from_bronze(
    spark: "SparkSession",
    bronze_table_name: str,
    training_table_name: str,
) -> None:
    """Create/refresh training table from bronze raw events.

    This keeps the example simple: we select canonical columns and add a label_ts.
    In production, you'd typically include richer cleaning and quality checks.
    """

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {training_table_name}
        USING DELTA
        AS
        SELECT
            customer_id,
            event_ts,
            CAST(event_ts AS timestamp) AS label_ts,
            amount,
            region,
            label
        FROM {bronze_table_name}
        """
    )


def summarize_events(events: Iterable[dict]) -> dict:
    """Small pure-Python helper useful for unit tests and sanity checks."""

    events = list(events)
    if not events:
        return {"count": 0, "positive_labels": 0}

    return {
        "count": len(events),
        "positive_labels": sum(int(e["label"]) for e in events),
    }
