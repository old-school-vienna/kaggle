import os
import time
import traceback as tb
from dataclasses import dataclass
from pprint import pprint
from typing import Optional, Callable

import pyspark.sql.functions as sfunc
from pyspark.ml import Estimator, Transformer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GeneralizedLinearRegression, GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel
from pyspark.sql import DataFrame, SparkSession

import common as cm
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
    param_grid: Callable[[Estimator], list]


@dataclass
class EstiResult:
    esti: Esti
    pip_result: PipResult


def param_grid_lr(esti: Estimator) -> list:
    return ParamGridBuilder() \
        .addGrid(esti.regParam, [0.1, 0.01]) \
        .addGrid(esti.fitIntercept, [False, True]) \
        .build()


def param_grid_gbtr(esti: Estimator) -> list:
    return ParamGridBuilder() \
        .addGrid(esti.maxBins, [16, 32]) \
        .addGrid(esti.maxDepth, [5, 10]) \
        .build()


def param_grid_empty(esti: Estimator) -> list:
    return ParamGridBuilder() \
        .build()


def train(data: DataFrame, esti: Estimator, eid: str, param_grid_builder: Callable[[Estimator], list]) -> PipResult:
    try:
        print(f"--- train {eid}")
        # Prepare training and test data.
        df_train, df_test = data.randomSplit([0.9, 0.1], seed=12345)

        # We use a ParamGridBuilder to construct a grid of parameters to search over.
        # TrainValidationSplit will try all combinations of values and determine best model using
        # the evaluator.
        params = param_grid_builder(esti)
        print(f"--- params")
        pprint(params)

        # In this case the estimator is simply the linear regression.
        # A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
        tvs = TrainValidationSplit(estimator=esti,
                                   estimatorParamMaps=params,
                                   evaluator=RegressionEvaluator(),
                                   # 80% of the data will be used for training, 20% for validation.
                                   trainRatio=0.8)

        # Run TrainValidationSplit, and choose the best set of parameters.
        trained_models: TrainValidationSplitModel = tvs.fit(df_train)

        # Make predictions on test data. model is the model with combination of parameters
        # that performed best.
        predictions = trained_models.transform(df_test) \
            .select("features", "label", "prediction")

        # Select (prediction, true label) and compute test error
        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)

        print(f"-- Root Mean Squared Error (RMSE) on test data = {rmse}")
        fnam = cm.fnam(eid)
        hlp.save_model(trained_models.bestModel, hlp.get_datadir(), fnam)
        print(f"-- saved model to {fnam}")
        return PipResult(rmse, trained_models.bestModel, "OK")
    except Exception:
        print(tb.format_exc())
        return PipResult(0.0, None, "ERROR")


def train_simple(data: DataFrame, esti: Estimator, eid: str) -> PipResult:
    """
    Train without cross validation
    """
    try:
        print(f"--- train_simple {eid}")
        # Prepare training and test data.
        df_train, df_test = data.randomSplit([0.9, 0.1], seed=12345)

        # Run TrainValidationSplit, and choose the best set of parameters.
        trained_model: Transformer = esti.fit(df_train)

        # Make predictions on test data. model is the model with combination of parameters
        # that performed best.
        predictions = trained_model.transform(df_test) \
            .select("features", "label", "prediction")

        # Select (prediction, true label) and compute test error
        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)

        print(f"-- Root Mean Squared Error (RMSE) on test data = {rmse}")
        fnam = cm.fnam(eid)
        hlp.save_model(trained_model, hlp.get_datadir(), fnam)
        print(f"-- saved model to {fnam}")
        return PipResult(rmse, trained_model, "OK")
    except Exception:
        print(tb.format_exc())
        return PipResult(0.0, None, "ERROR")


def main():
    start = time.time()
    spark = SparkSession.builder \
        .appName(os.path.basename(__file__)) \
        .getOrCreate()
    df_name = "s5_01_small"
    print(f"-- df: {df_name}")
    pp: DataFrame = hlp.readFromDatadirParquet(spark, df_name) \
        .where(sfunc.col("label").isNotNull())

    estis = [
        Esti(pp, GeneralizedLinearRegression(family='gaussian', link='identity'),
             "glr gaussian identity", "glrgi", param_grid_lr),
        Esti(pp, GBTRegressor(),
             "gradient boost regressor", "gbtr", param_grid_gbtr),
    ]
    estis_all = [
        Esti(pp, GeneralizedLinearRegression(family='poisson', link='identity'),
             "glr poisson identity", "glrpi", param_grid_lr),
        Esti(pp, GeneralizedLinearRegression(family='gaussian', link='identity'),
             "glr gaussian identity", "glrgi", param_grid_lr),
        Esti(pp, GeneralizedLinearRegression(family='poisson', link='log'),
             "glr poisson log", "glrpi", param_grid_lr),
        Esti(pp, GBTRegressor(),
             "gradient boost regressor", "gbtr", param_grid_gbtr),
    ]

    print(f"-- training {len(estis)} estimators")
    reses = [EstiResult(e, train_simple(e.data, e.esti, e.id)) for e in estis]
    
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
    main()
