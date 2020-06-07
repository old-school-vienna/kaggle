from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

from pyspark import SparkContext
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession

import helpers as hlp
from common import fnam


def no_idea():
    spark = SparkSession.builder \
        .appName(__name__) \
        .getOrCreate()

    sc: SparkContext = spark.sparkContext

    rdd_long = sc.parallelize(range(0, 10000000))

    short = rdd_long.take(20)
    print(f"type from take: {type(short)}")

    rdd_short = sc.parallelize(short) \
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
    path = Path("target")
    if not path.exists():
        path.mkdir()
    nam = "gr1.sp"

    def save_load_hlp():
        SparkSession.builder \
            .appName("tryout") \
            .getOrCreate()
        m = LinearRegression(regParam=0.5, maxIter=10)
        pm = m.extractParamMap()
        pprint(pm)
        print()
        pprint(str(path))
        hlp.save_model(m, path, nam)

        m2 = hlp.load_model(path, nam)
        pm2 = m2.extractParamMap()
        pprint(pm2)

    def load_hlp():
        SparkSession.builder \
            .appName("tryout") \
            .getOrCreate()

        m2 = hlp.load_model(path, nam)
        pm2 = m2.extractParamMap()
        pprint(pm2)

    save_load_hlp()


def load_model():
    SparkSession.builder \
        .appName("tryout") \
        .getOrCreate()
    eid = "glrgi"
    fn = fnam(eid)
    m = hlp.load_model(hlp.get_datadir(), fn)
    print(f"----reading {fn}----------")
    pprint(m.extractParamMap())


load_model()
