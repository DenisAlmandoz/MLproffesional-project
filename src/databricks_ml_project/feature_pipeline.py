"""Feature engineering patterns for Databricks.

This module demonstrates:
1) Point-in-time correct feature lookups (to prevent leakage).
2) Batch feature computation pipelines.
3) Streaming feature aggregation for near-real-time use cases.
4) A scaffold for preparing online-ready tables.

The functions are intentionally explicit and verbose so exam preparation is easy.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def build_batch_features(events_df: DataFrame) -> DataFrame:
    """Create stable customer-level aggregate features from historical events.

    Why this matters:
    - Batch aggregates are common in SparkML training pipelines.
    - This function demonstrates deterministic feature creation suitable for
      offline training tables and repeatable backfills.
    """

    return (
        events_df.groupBy("customer_id")
        .agg(
            F.count("*").alias("event_count_30d"),
            F.avg("amount").alias("avg_amount_30d"),
            F.max("event_ts").alias("latest_event_ts"),
        )
        .fillna({"avg_amount_30d": 0.0})
    )


def point_in_time_join(labels_df: DataFrame, features_history_df: DataFrame) -> DataFrame:
    """Perform point-in-time correct join between labels and feature history.

    The rule: for each label timestamp, only use features generated at or before
    that timestamp. This avoids leaking future information into training.
    """

    joined = (
        labels_df.alias("l")
        .join(
            features_history_df.alias("f"),
            on=(F.col("l.customer_id") == F.col("f.customer_id"))
            & (F.col("f.feature_ts") <= F.col("l.label_ts")),
            how="left",
        )
        .select(
            F.col("l.*"),
            F.col("f.feature_ts"),
            F.col("f.event_count_30d"),
            F.col("f.avg_amount_30d"),
        )
    )

    latest_feature_window = Window.partitionBy("customer_id", "label_ts").orderBy(F.col("feature_ts").desc())

    return (
        joined.withColumn("feature_rank", F.row_number().over(latest_feature_window))
        .filter(F.col("feature_rank") == 1)
        .drop("feature_rank", "feature_ts")
    )


def build_streaming_features(spark: SparkSession, source_table: str, target_table: str) -> None:
    """Create a structured streaming aggregation pipeline.

    This pattern is useful when features need to refresh continuously. In many
    production systems, this table feeds both monitoring and low-latency serving.
    """

    source_df = spark.readStream.table(source_table)

    aggregated = (
        source_df.withWatermark("event_ts", "30 minutes")
        .groupBy(
            F.col("customer_id"),
            F.window("event_ts", "15 minutes", "5 minutes").alias("time_window"),
        )
        .agg(
            F.count("*").alias("events_15m"),
            F.sum("amount").alias("sum_amount_15m"),
        )
        .select(
            "customer_id",
            F.col("time_window.end").alias("feature_ts"),
            "events_15m",
            "sum_amount_15m",
        )
    )

    (
        aggregated.writeStream
        .option("checkpointLocation", f"/tmp/checkpoints/{target_table.replace('.', '_')}")
        .trigger(processingTime="5 minutes")
        .toTable(target_table)
    )


def prepare_online_table_note() -> str:
    """Return guidance string for online table configuration with Databricks SDK.

    The exact SDK call shape can vary by runtime and workspace settings, so this
    project provides the deploy intent explicitly for adaptation in your workspace.
    """

    return (
        "Use Databricks SDK WorkspaceClient().online_tables.create(...) to map a "
        "Unity Catalog feature table to an online endpoint for low-latency lookups."
    )
