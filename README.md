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
- `scripts/`: thin orchestration entry points for Databricks jobs
- `dab/`: Databricks Asset Bundle configuration scaffold
- `tests/`: unit + integration-style tests for core logic

## How to Use in Databricks

1. Import this repository into Databricks Repos.
2. Create a Databricks Asset Bundle deployment target using `dab/databricks.yml`.
3. Attach a Databricks ML runtime cluster (or serverless workflows where supported).
4. Run orchestration scripts as Databricks Jobs tasks:
   - `scripts/run_feature_pipeline.py`
   - `scripts/run_training.py`
   - `scripts/run_monitoring.py`
   - `scripts/run_retraining.py`
5. Configure secrets and Unity Catalog names via environment variables.

## Environment Variables

- `DBX_CATALOG` (default: `main`)
- `DBX_SCHEMA` (default: `ml_exam`)
- `MLFLOW_EXPERIMENT_NAME` (default: `/Shared/ml_exam_project`)
- `MODEL_NAME` (default: `main.ml_exam.exam_candidate_model`)

## Notes

- The code is heavily documented so each module explains *why* and *how* to implement the corresponding exam concepts.
- Some APIs (Lakehouse Monitoring, Feature Engineering online serving, endpoint traffic config) are represented with practical scaffolds because exact payloads can vary by workspace/runtime version.
