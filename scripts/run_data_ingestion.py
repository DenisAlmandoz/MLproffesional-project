"""Databricks Job entry point for data generation + ingestion.

This script makes the project self-contained by creating synthetic data,
ingesting it into a bronze Delta table, and initializing a training table.
"""

from pyspark.sql import SparkSession

from databricks_ml_project.config import CONFIG
from databricks_ml_project.data_ingestion import (
    ensure_training_table_from_bronze,
    generate_synthetic_events_csv,
    ingest_to_bronze_table,
    read_raw_events_csv,
)


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()

    csv_path = generate_synthetic_events_csv(
        output_path=CONFIG.local_seed_data_path,
        n_rows=3000,
        seed=123,
    )
    print(f"Generated seed data at: {csv_path}")

    events_df = read_raw_events_csv(spark, csv_path)
    ingest_to_bronze_table(events_df, CONFIG.bronze_events_table)
    ensure_training_table_from_bronze(spark, CONFIG.bronze_events_table, CONFIG.training_table)

    print(f"Bronze table ready: {CONFIG.bronze_events_table}")
    print(f"Training table ready: {CONFIG.training_table}")
