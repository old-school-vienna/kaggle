import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Optional, Callable
import traceback as tb

import pyspark.sql.functions as sfunc
import pyspark.sql.types as stype
from pyspark.ml import Pipeline, Estimator, Transformer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression, RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel
from pyspark.sql import DataFrame, SparkSession

import helpers as hlp


@dataclass
class PipResult:
    rmse: float
    model: Optional[Estimator]
    train_state: str


@dataclass
class Esti:
    data: DataFrame
    esti: Estimator
    desc: str
    id: str
    train_call: Callable[[DataFrame, Estimator], PipResult]


@dataclass
class EstiResult:
    esti: Esti
    pip_result: PipResult


def train_dummy(pp: DataFrame, esti: Estimator) -> PipResult:
    glr = GeneralizedLinearRegression()
    return PipResult(7.9238429847, glr, 'OK')


def train_lr(data: DataFrame, esti: Estimator) -> PipResult:
    try:
        # Prepare training and test data.
        train, test = data.randomSplit([0.9, 0.1], seed=12345)

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
        besti: TrainValidationSplitModel = tvs.fit(train)

        # Make predictions on test data. model is the model with combination of parameters
        # that performed best.
        predictions = besti.transform(test) \
            .select("features", "label", "prediction")

        # Select (prediction, true label) and compute test error
        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)

        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
        return PipResult(rmse, besti.bestModel, "OK")
    except Exception:
        print(tb.format_exc())
        return PipResult(0.0, None, "ERROR")


def main():
    start = time.time()
    spark = SparkSession.builder \
        .appName(os.path.basename(__file__)) \
        .getOrCreate()
    pp: DataFrame = hlp.read(spark, hlp.get_datadir(), "s5_01") \
        .where(sfunc.col("label").isNotNull())
    estis = [
        Esti(pp, GeneralizedLinearRegression(family='gaussian', link='identity', maxIter=10, regParam=0.3),
             "glr gaussian identity", "glrgi", train_lr),
    ]
    estis_glr = [
        Esti(pp, GeneralizedLinearRegression(family='poisson', link='identity', maxIter=10, regParam=0.3),
             "glr poisson identity", "glrpi", train_lr),
        Esti(pp, GeneralizedLinearRegression(family='gaussian', link='identity', maxIter=10, regParam=0.3),
             "glr gaussian identity", "glrgi", train_lr),
        Esti(pp, GeneralizedLinearRegression(family='poisson', link='log', maxIter=10, regParam=0.3),
             "glr poisson log", "glrpi", train_lr),
    ]
    estis_class = [
        Esti(pp, GBTClassifier(),
             "gbt default", "gbtd", train_dummy),
        Esti(pp, RandomForestRegressor(),
             "rf default", "rfd", train_dummy),
    ]
    reses = [EstiResult(e, e.train_call(e.data, e.esti)) for e in estis]
    print()
    print(f"+-----------------------------------------------------------+")
    print(f"|model                                |mse       |state     |")
    print(f"|-------------------------------------+----------+----------|")
    for res in reses:
        print(f"|{res.esti.desc:37}|{res.pip_result.rmse:10.4f}|{res.pip_result.train_state:10}|")
    print(f"+-----------------------------------------------------------+")
    print()
    for res in reses:
        print(f"Hyperparams: {res.esti.desc}")
        model = res.pip_result.model
        if model is None:
            print(f"Model for {res.esti.desc} could not be created.")
        else:
            print(f"+-------model-------------------------------------+")
            pprint(model)
            print(f"+-------params------------------------------------+")
            pprint(model.extractParamMap())
            print()
    end = time.time()
    elapsed = end - start
    elapseds = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"------------------------- R E A D Y ------------ {elapseds} --------------------")
    spark.stop()


if __name__ == '__main__':
    # preprocessing()
    main()
