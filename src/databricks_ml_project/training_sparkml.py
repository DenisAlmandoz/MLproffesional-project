"""SparkML training and scoring patterns.

This module covers core exam skills:
- Building an ML pipeline with transformers + estimator.
- Hyperparameter tuning with CrossValidator.
- Evaluation metrics.
- Batch and streaming inference selection patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame


@dataclass
class SparkMLArtifacts:
    model: object
    auc: float


def train_sparkml_pipeline(train_df: DataFrame) -> SparkMLArtifacts:
    """Train and tune a SparkML classifier using distributed execution."""

    indexer = StringIndexer(inputCol="region", outputCol="region_idx", handleInvalid="keep")
    encoder = OneHotEncoder(inputCols=["region_idx"], outputCols=["region_ohe"])

    assembler = VectorAssembler(
        inputCols=["event_count_30d", "avg_amount_30d", "region_ohe"],
        outputCol="features",
    )

    estimator = GBTClassifier(labelCol="label", featuresCol="features", maxIter=30)

    pipeline = Pipeline(stages=[indexer, encoder, assembler, estimator])

    param_grid = (
        ParamGridBuilder()
        .addGrid(estimator.maxDepth, [3, 5])
        .addGrid(estimator.stepSize, [0.05, 0.1])
        .build()
    )

    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4,
    )

    model = cv.fit(train_df)
    auc = evaluator.evaluate(model.transform(train_df))

    return SparkMLArtifacts(model=model, auc=auc)


def score_batch(model, batch_df: DataFrame) -> DataFrame:
    """Score a batch DataFrame and expose business-friendly prediction columns."""

    scored = model.transform(batch_df)
    return scored.select("customer_id", "probability", "prediction")


def score_streaming(model, streaming_df: DataFrame) -> DataFrame:
    """Score a streaming DataFrame.

    In production, the output could be sent to Delta, Kafka, or a serving tier.
    """

    return model.transform(streaming_df).select("customer_id", "prediction")
