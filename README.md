# Databricks End-to-End ML Lifecycle Project

This repository provides a **production-style Databricks machine learning project** that maps directly to the advanced topics from the Databricks ML certification outline:

- SparkML model development and scoring (batch + streaming)
- Distributed training and hyperparameter tuning (Optuna + Ray)
- Advanced MLflow usage (nested runs, custom artifacts/metrics)
- Feature engineering with point-in-time correctness and online feature readiness
- MLOps architecture (unit/integration testing, environment promotion, DAB assets)
- Automated retraining on drift/performance signals
- Lakehouse Monitoring and alert-friendly drift metrics
- Model deployment patterns (custom pyfunc, canary/blue-green strategy helpers)

## Project Layout

- `src/databricks_ml_project/`: core implementation modules
- `scripts/`: orchestration entry points for Databricks jobs
- `dab/`: Databricks Asset Bundle configuration scaffold
- `tests/`: unit + integration-style tests for core logic

## Built-in Data Ingestion (Yes, data is included now)

This project now includes built-in synthetic data generation and ingestion so you can run it immediately in Databricks:

- `src/databricks_ml_project/data_ingestion.py`
  - Generates reproducible synthetic events CSV with `customer_id`, `event_ts`, `amount`, `region`, and `label`.
  - Reads typed raw CSV into Spark.
  - Ingests into Delta bronze table.
  - Initializes training table from bronze.
- `scripts/run_data_ingestion.py`
  - Job entrypoint that runs the full data bootstrap.

Default generated raw file location:
- `/dbfs/tmp/ml_exam/raw/events.csv`

Default Delta tables:
- `main.ml_exam.bronze_events`
- `main.ml_exam.training_events`

## How to Use in Databricks

1. Import this repository into Databricks Repos.
2. Create catalog/schema (if missing), then deploy bundle from `dab/databricks.yml`.
3. Run `scripts/run_data_ingestion.py` first to generate and ingest seed data.
4. Run the remaining orchestration scripts as Databricks Jobs tasks:
   - `scripts/run_feature_pipeline.py`
   - `scripts/run_training.py`
   - `scripts/run_monitoring.py`
   - `scripts/run_retraining.py`

## Environment Variables

- `DBX_CATALOG` (default: `main`)
- `DBX_SCHEMA` (default: `ml_exam`)
- `MLFLOW_EXPERIMENT_NAME` (default: `/Shared/ml_exam_project`)
- `MODEL_NAME` (default: `main.ml_exam.exam_candidate_model`)
- `SEED_DATA_PATH` (default: `/dbfs/tmp/ml_exam/raw/events.csv`)

## Notes

- The code is heavily documented so each module explains *why* and *how* to implement the corresponding exam concepts.
- Some APIs (Lakehouse Monitoring, Feature Engineering online serving, endpoint traffic config) are represented with practical scaffolds because exact payloads can vary by workspace/runtime version.
