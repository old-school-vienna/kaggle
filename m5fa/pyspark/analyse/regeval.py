import os
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("regval") \
    .getOrCreate()

datadir: str = os.getenv("DATADIR")
if datadir is None:
    raise ValueError("Environment variable DATADIR must be defined")
print(f"datadir = '{datadir}'")

p = Path(datadir, "s5_01.parquet")
print(f"Reading: '{p}'")

# Load and parse the data file, converting it to a DataFrame.
data = spark.read.parquet(str(p)) \
    .filter("label is not null")

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a GBT model.
gbt = GBTRegressor(maxIter=10)

# Chain indexer and GBT in a Pipeline
pipeline = Pipeline(stages=[gbt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

gbtModel = model.stages[1]
print(gbtModel)  # summary only
