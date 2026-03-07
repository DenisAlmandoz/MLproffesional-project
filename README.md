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
2. Create or confirm Unity Catalog objects exist:
   - Catalog: `main` (or your `DBX_CATALOG` value)
   - Schema: `ml_exam` (or your `DBX_SCHEMA` value)
3. Set environment variables for your workspace/job (optional if using defaults):
   - `DBX_CATALOG`, `DBX_SCHEMA`, `MLFLOW_EXPERIMENT_NAME`, `MODEL_NAME`, `SEED_DATA_PATH`

   Where to define these variables:
   - **Databricks Workflows Job UI (recommended):**
     - Jobs -> your job -> Edit task -> *Environment variables*.
     - Add key/value pairs per task (for example on `ingest_seed_data`, `train_model`, etc.).
     - Use the same values for every task that imports `CONFIG` from `src/databricks_ml_project/config.py`.

     Example values to add:
     - `DBX_CATALOG=main`
     - `DBX_SCHEMA=ml_exam`
     - `MLFLOW_EXPERIMENT_NAME=/Shared/ml_exam_project`
     - `MODEL_NAME=main.ml_exam.exam_candidate_model`
     - `SEED_DATA_PATH=/dbfs/tmp/ml_exam/raw/events.csv`
   - **Databricks notebook/script runtime (temporary):**
     - In a notebook cell before running scripts:
       - `import os`
       - `os.environ["DBX_CATALOG"] = "main"`
   - **Databricks Asset Bundle variables (deployment-time):**
     - Use `--var` flags with bundle commands, for example:
       - `databricks bundle deploy --target dev --var="catalog=main" --var="schema=ml_exam" --var="existing_cluster_id=<your-cluster-id>"`
     - This controls bundle resource names/values at deploy time.
   - **GitHub Actions (for CI/CD):**
     - Use repository/environment secrets for sensitive values (for example tokens), and workflow `env:` for non-sensitive defaults.

4. Deploy the Databricks Asset Bundle from `dab/databricks.yml`.


### Quick answer: where should these be defined?

If you are running this project as Databricks Jobs, define them in the **Job Task Environment Variables** section for each task. That is the primary and recommended location.

## End-to-End Run Order (What to run first, second, etc.)

Follow this exact order for a full lifecycle run:

### Step 0 — Bootstrap tables and raw data (run first)

Run:
- `scripts/run_data_ingestion.py`

What it does:
- Generates synthetic events CSV in DBFS (default `/dbfs/tmp/ml_exam/raw/events.csv`).
- Ingests raw events into bronze Delta table (`main.ml_exam.bronze_events`).
- Creates/initializes training table (`main.ml_exam.training_events`).

### Step 1 — Build features

Run:
- `scripts/run_feature_pipeline.py`

What it does:
- Prepares the feature engineering workflow scaffold and target feature table configuration.
- Provides point-in-time/online-feature guidance for productionizing the feature pipeline.

### Step 2 — Train and tune model

Run:
- `scripts/run_training.py`

What it does:
- Reads training-table/experiment configuration.
- Serves as training task entrypoint for SparkML + tracking modules in `src/databricks_ml_project/`.

### Step 3 — Run monitoring

Run:
- `scripts/run_monitoring.py`

What it does:
- Builds Lakehouse Monitoring payload for inference tables.
- Supports drift/alert workflows via monitoring utilities.

### Step 4 — Evaluate retraining trigger

Run:
- `scripts/run_retraining.py`

What it does:
- Demonstrates retraining decision logic based on drift/performance conditions.
- Supports model selection logic in retraining pipelines.

### Step 5 — Deploy and serve model (after validation)

Use deployment helpers in:
- `src/databricks_ml_project/deployment.py`
- `src/databricks_ml_project/mlflow_tracking.py`

What it does:
- Registers custom pyfunc models.
- Builds canary/blue-green rollout payloads for Databricks Model Serving.

## Suggested Job Dependency Order in Databricks Workflows

When creating a multi-task Databricks Job, define dependencies in this order:

1. `ingest_seed_data`
2. `build_features`
3. `train_model`
4. `monitor`
5. `retrain`

This ensures all downstream tasks have the required upstream data/assets.


## CI/CD Automation (GitHub Actions)

This repository now includes CI and CD workflows:

- `.github/workflows/ci.yml`
  - Runs on pull requests and pushes.
  - Installs dependencies, compiles Python files, and runs `pytest`.

- `.github/workflows/cd.yml`
  - Deploys Databricks Asset Bundle (`dab/databricks.yml`) to your workspace.
  - Triggered automatically on push to `main` or manually with `workflow_dispatch`.
  - Supports deployment targets `dev` and `prod`.


### CD troubleshooting

- If `databricks bundle validate --target dev` fails with `default auth: cannot configure default credentials`, configure auth secrets in GitHub (`DATABRICKS_TOKEN` or `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET`) and rerun.
- CD runs bundle commands from `dab/`, so the bundle root is `dab/databricks.yml`.
- If you see `Provided PAT token does not have required scopes: scim`, switch CD auth to OAuth M2M (`DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET`) or use a PAT with the required scopes in your workspace.
- If you see `path .../scripts/... is not contained in sync root path`, ensure the bundle includes repo-level sync paths. This project sets `sync.paths` in `dab/databricks.yml` to include `../scripts`, `../src`, and `../pyproject.toml`.
- If deploy fails with `Cluster ... does not exist`, set `DATABRICKS_CLUSTER_ID` to a real, running cluster ID in your target workspace.
- If deploy fails with `CreateRegisteredModel name ... is not a valid name`, use a workspace-model-safe name (no dots); this bundle uses `exam_candidate_model`.
- If validation fails with `Task ... requires a cluster or an environment to run`, provide `existing_cluster_id` via bundle variables (for example `--var="existing_cluster_id=<your-cluster-id>"`). This workflow passes it from `DATABRICKS_CLUSTER_ID`.

### Required GitHub Secrets for CD

Set these repository secrets before running CD:

- `DATABRICKS_HOST` (for example: `https://dbc-xxxxxxxx.cloud.databricks.com`)
- `DATABRICKS_CLUSTER_ID` (required for this bundle; existing cluster used by job tasks)
- Authentication (choose one):
  - `DATABRICKS_TOKEN` (PAT; must include required scopes in your workspace)
  - `DATABRICKS_CLIENT_ID` + `DATABRICKS_CLIENT_SECRET` (OAuth M2M, recommended)

### Recommended release flow

1. Open PR -> CI runs (`compileall` + `pytest`).
2. Merge to `main` after CI passes.
3. CD automatically validates and deploys bundle to `dev` target by default.
4. Use manual dispatch to deploy to `prod` when ready.

## Environment Variables

- `DBX_CATALOG` (default: `main`)
- `DBX_SCHEMA` (default: `ml_exam`)
- `MLFLOW_EXPERIMENT_NAME` (default: `/Shared/ml_exam_project`)
- `MODEL_NAME` (default: `main.ml_exam.exam_candidate_model`)
- `SEED_DATA_PATH` (default: `/dbfs/tmp/ml_exam/raw/events.csv`)
- `existing_cluster_id` (bundle variable; default placeholder in `dab/databricks.yml`, override at deploy time)

## Notes

- The code is heavily documented so each module explains *why* and *how* to implement the corresponding exam concepts.
- Some APIs (Lakehouse Monitoring, Feature Engineering online serving, endpoint traffic config) are represented with practical scaffolds because exact payloads can vary by workspace/runtime version.
