import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyspark.sql.functions as sfunc
import pyspark.sql.types as stype
from pyspark.ml import Pipeline, Estimator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import DataFrame, SparkSession


@dataclass
class Esti:
    data: DataFrame
    esti: Estimator
    desc: str
    id: str


@dataclass
class PipResult:
    rmse: float
    hparams: dict


@dataclass
class EstiResult:
    esti: Esti
    pip_result: PipResult


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

    schema = stype.StructType([
        stype.StructField('year', stype.IntegerType(), True),
        stype.StructField('month', stype.IntegerType(), True),
        stype.StructField('dn', stype.IntegerType(), True),
        stype.StructField('wday', stype.IntegerType(), True),
        stype.StructField('snap', stype.IntegerType(), True),
        stype.StructField('dept_id', stype.StringType(), True),
        stype.StructField('item_id', stype.StringType(), True),
        stype.StructField('store_id', stype.StringType(), True),
        stype.StructField('sales', stype.DoubleType(), True),
        stype.StructField('flag_ram', stype.IntegerType(), True),
        stype.StructField('Sales_Pred', stype.DoubleType(), True)
    ])

    csv_path = datadir / "Sales5_Ab2011_InklPred.csv"
    print(f"--- Reading: '{csv_path}'")

    sales5: DataFrame = spark.read.csv(str(csv_path), header='true', schema=schema) \
        .withColumn("label", sfunc.col('sales'))

    ppdf = prepro(sales5)
    print(f"--- Writing: '{pppath}'")
    ppdf.write \
        .format("parquet") \
        .mode("overwrite") \
        .save(str(pppath))


def pipeline1(pp: DataFrame, esti: Estimator) -> PipResult:
    marams = {
        'p1': 298347,
        'hste_la_vista': 'H',
        'good': True,
        'value_up': 8.2347,
        'nix': None
    }
    return PipResult(7.9238429847, marams)


def pipeline(pp: DataFrame, esti: Estimator) -> PipResult:
    # Prepare training and test data.
    train, test = pp.randomSplit([0.9, 0.1], seed=12345)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # TrainValidationSplit will try all combinations of values and determine best model using
    # the evaluator.
    param_grid = ParamGridBuilder() \
        .addGrid(esti.regParam, [0.1, 0.01]) \
        .addGrid(esti.fitIntercept, [False, True]) \
        .build()

    # In this case the estimator is simply the linear regression.
    # A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    tvs = TrainValidationSplit(estimator=esti,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    # Run TrainValidationSplit, and choose the best set of parameters.
    besti = tvs.fit(train)
    param_map = besti.bestModel.extractParamMap()

    # Make predictions on test data. model is the model with combination of parameters
    # that performed best.
    predictions = besti.transform(test) \
        .select("features", "label", "prediction")

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    return PipResult(rmse, param_map)


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
        .appName(os.path.basename(__file__)) \
        .getOrCreate()
    pp: DataFrame = read_data(spark)
    estis1 = [
        Esti(pp, GeneralizedLinearRegression(family='poisson', link='identity', maxIter=10, regParam=0.3),
             "glr poisson identity", "glrpi"),
    ]
    estis2 = [
        Esti(pp, GeneralizedLinearRegression(family='gaussian', link='identity', maxIter=10, regParam=0.3),
             "glr gaussian identity", "glrgi"),
        Esti(pp, GeneralizedLinearRegression(family='poisson', link='log', maxIter=10, regParam=0.3),
             "glr poisson log", "glrpi"),
        Esti(pp, GeneralizedLinearRegression(family='binomial', link='logit', maxIter=10, regParam=0.3),
             "glr gamma identity", "glrgai"),
    ]
    reses = [EstiResult(e, pipeline1(e.data, e.esti)) for e in estis1]
    print()
    print(f"+------------------------------------------------+")
    print(f"|model                                |mse       |")
    print(f"|------------------------------------------------|")
    for res in reses:
        print(f"|{res.esti.desc:37}|{res.pip_result.rmse:10.4f}|")
    print(f"+------------------------------------------------+")
    print()
    for res in reses:
        print(f"Hyperparams: {res.esti.desc}")
        hpar: dict = res.pip_result.hparams
        print(f"+------------------------------------------------+")
        for k in hpar.keys():
            v = hpar[k]
            print(f"|{k:37}|{fval(v, 10)}|")
            print(f"+------------------------------------------------+")
        print()
    print()
    end = time.time()
    elapsed = end - start
    elapseds = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"------------------------- R E A D Y ------------ {elapseds} --------------------")
    spark.stop()
    exit(0)


def fval(value: Any, leng: int) -> str:
    if value is None:
        fstr = f"{{:{leng}}}"
        return fstr.format('None')
    if isinstance(value, int):
        fstr = f"{{:{leng}d}}"
        return fstr.format(value)
    if isinstance(value, float):
        fstr = f"{{:{leng}.3f}}"
        return fstr.format(value)
    else:
        fstr = f"{{:{leng}}}"
        return fstr.format(str(value))


if __name__ == '__main__':
    main()
