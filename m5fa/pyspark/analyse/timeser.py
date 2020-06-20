import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml import Transformer
from pyspark.ml.linalg import Vector, Vectors
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf

import helpers as hlp


def predict(features: Vector) -> float:
    # print_vec(features)
    return 1.345


class TTrans(Transformer):

    def _transform(self, dataset: DataFrame):
        f = udf(predict, returnType=T.FloatType())
        return dataset.withColumn("prediction", f(F.col('features')))


def print_vec(v0: Vector):
    def fmt(v: float) -> str:
        if v == 0:
            return '.'
        else:
            return f"|{v:.3f}|"

    v1 = v0.toArray()
    vs = ''.join([fmt(v) for v in v1])
    print(vs)


def set_at(vec: Vector, pos: int, values: list) -> Vector:
    vals = vec.toArray()
    for i in range(0, len(values)):
        vals[pos + i] = values[i]
    return Vectors.dense(vals)


def test_insert():
    spark = SparkSession.builder.getOrCreate()
    df: DataFrame = hlp.readFromDatadirParquet(spark, 's5_01_small') \
        .where('label is null')
    for r in df.collect():
        insert = [1.0, 2.2, 3.3]
        v0: Vector = r.features
        print_vec(v0)
        v1 = set_at(v0, 2, insert)
        print_vec(v1)


def test_ttrans():
    spark = SparkSession.builder.getOrCreate()
    df: DataFrame = hlp.readFromDatadirParquet(spark, 's5_01_small') \
        .where('label is null')
    ttrans = TTrans()
    df1 = ttrans.transform(df)
    df1.show()


test_ttrans()
