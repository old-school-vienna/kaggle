from dataclasses import dataclass
from pprint import pprint

from pyspark import SparkContext
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql import SparkSession


def no_idea():
    spark = SparkSession.builder \
        .appName(__name__) \
        .getOrCreate()

    sc: SparkContext = spark.sparkContext

    rdd_long = sc.parallelize(range(0, 10000000))

    l = rdd_long.take(20)
    print(f"type from take: {type(l)}")

    rdd_short = sc.parallelize(l) \
        .filter(lambda x: x % 2 == 0)

    print(f"len of short: {rdd_short.collect()}")


def dataclass_example():
    @dataclass
    class X:
        nam: str
        cnt: int

    x = X(nam='hallo', cnt=11)

    print(x)


def save_model():
    def save_load():
        path = "/tmp/m1.sp"
        spark = SparkSession.builder \
            .appName("tryout") \
            .getOrCreate()
        m = GeneralizedLinearRegression(regParam=0.5, maxIter=10)
        pm = m.extractParamMap()
        pprint(pm)
        m.save(path)

        m2c = eval("GeneralizedLinearRegression")
        m2 = m2c.load(path)
        pm2 = m2.extractParamMap()
        pprint(pm2)

    save_load()

save_model()
