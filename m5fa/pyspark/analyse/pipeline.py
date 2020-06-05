import os
import time
from pathlib import Path

import pyspark.sql.types as t
import pyspark.sql.functions as f
from pyspark.ml import Pipeline, Estimator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


def preprocessing(spark: SparkSession, pppath: Path, datadir: Path):
    def prepro(s5: DataFrame) -> DataFrame:
        stages = []
        catvars = ['dept_id', 'item_id', 'store_id', 'wday']
        for v in catvars:
            stages += [StringIndexer(inputCol=v,
                                     outputCol=f"i{v}")]
        stages += [OneHotEncoderEstimator(inputCols=[f"i{v}" for v in catvars],
                                          outputCols=[f"v{v}" for v in catvars])]
        stages += [VectorAssembler(inputCols=['vwday', 'vitem_id', 'vdept_id', 'vstore_id', 'flag_ram',
                                              'snap', 'dn', 'month', 'year'],
                                   outputCol='features')]

        pip: Pipeline = Pipeline(stages=stages)
        pipm = pip.fit(s5)
        df: DataFrame = pipm.transform(s5)
        return df.drop('idept_id', 'iitem_id', 'istore_id', 'iwday', 'vdept_id', 'vtem_id', 'vstore_id', 'vwday')

    print("--- preprocessing -----------------------")

    schema = t.StructType([
        t.StructField('year', t.IntegerType(), True),
        t.StructField('month', t.IntegerType(), True),
        t.StructField('dn', t.IntegerType(), True),
        t.StructField('wday', t.IntegerType(), True),
        t.StructField('snap', t.IntegerType(), True),
        t.StructField('dept_id', t.StringType(), True),
        t.StructField('item_id', t.StringType(), True),
        t.StructField('store_id', t.StringType(), True),
        t.StructField('sales', t.DoubleType(), True),
        t.StructField('flag_ram', t.IntegerType(), True),
        t.StructField('Sales_Pred', t.DoubleType(), True)
    ])

    csv_path = datadir / "Sales5_Ab2011_InklPred.csv"
    print(f"--- Reading: '{csv_path}'")

    sales5: DataFrame = spark.read.csv(str(csv_path), header='true', schema=schema) \
        .withColumn("label", f.col('sales'))

    ppdf = prepro(sales5)
    print(f"--- Writing: '{pppath}'")
    ppdf.write \
        .format("parquet") \
        .mode("overwrite") \
        .save(str(pppath))


def pipeline1(pp: DataFrame, esti: Estimator) -> float:
    return 0.6475


def pipeline(pp: DataFrame, esti: Estimator) -> float:
    # Prepare training and test data.
    train, test = pp.randomSplit([0.9, 0.1], seed=12345)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # TrainValidationSplit will try all combinations of values and determine best model using
    # the evaluator.
    param_grid = ParamGridBuilder() \
        .addGrid(esti.regParam, [0.1, 0.01]) \
        .addGrid(esti.fitIntercept, [False, True]) \
        .addGrid(esti.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    # In this case the estimator is simply the linear regression.
    # A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    tvs = TrainValidationSplit(estimator=esti,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    # Run TrainValidationSplit, and choose the best set of parameters.
    esti = tvs.fit(train)

    # Make predictions on test data. model is the model with combination of parameters
    # that performed best.
    predictions = esti.transform(test) \
        .select("features", "label", "prediction")

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    return rmse


def read_data(spark: SparkSession) -> DataFrame:
    datadir: Path = Path(os.getenv("DATADIR"))
    if datadir is None:
        raise ValueError("Environment variable DATADIR must be defined")
    print(f"datadir = '{datadir}'")

    ppnam = "s5_01"
    pppath = datadir / f"{ppnam}.parquet"
    if not pppath.exists():
        preprocessing(spark, pppath, datadir)
    print(f"--- Reading: '{pppath}'")
    return spark.read.parquet(str(pppath)) \
        .filter("label is not null")


def main():
    start = time.time()
    spark = SparkSession.builder \
        .appName(__name__) \
        .getOrCreate()
    pp: DataFrame = read_data(spark)
    estis = [(pp, LinearRegression(maxIter=10), "lr max iter 10")]
    mses = [(pipeline1(t[0], t[1]), t[2]) for t in estis]
    print(mses)
    end = time.time()
    elapsed = end - start
    elapseds = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"------------------------- R E A D Y ------------ {elapseds} --------------------")
    spark.stop()
    exit(0)


if __name__ == '__main__':
    main()
