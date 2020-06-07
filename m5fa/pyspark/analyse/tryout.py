from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import pyspark.sql.types as t
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


def submission():
    testval = [
        [1, 'HOBBIES_1_001', 'CA_1', 2.2],
        [1, 'HOBBIES_1_001', 'CA_2', 0.0],
        [1, 'HOBBIES_1_001', 'TX_1', 2.0],
        [1, 'HOBBIES_1_002', 'CA_1', 2.0],
        [1, 'HOBBIES_1_002', 'CA_2', 0.0],
        [1, 'HOBBIES_1_002', 'TX_1', 2.0],
        [2, 'HOBBIES_1_001', 'CA_1', 2.0],
        [2, 'HOBBIES_1_001', 'CA_2', 0.0],
        [2, 'HOBBIES_1_001', 'TX_1', 2.0],
        [2, 'HOBBIES_1_001', 'TX_2', 1.0],
        [2, 'HOBBIES_1_002', 'CA_1', 0.0],
        [2, 'HOBBIES_1_002', 'CA_2', 0.0],
        [2, 'HOBBIES_1_002', 'TX_1', 2.0],
        [3, 'HOBBIES_1_001', 'CA_1', 2.0],
        [3, 'HOBBIES_1_001', 'CA_2', 0.0],
        [3, 'HOBBIES_1_001', 'TX_1', 2.0],
        [3, 'HOBBIES_1_002', 'CA_1', 2.2],
        [3, 'HOBBIES_1_002', 'CA_2', 0.3],
        [3, 'HOBBIES_1_002', 'TX_1', 2.4],
        [4, 'HOBBIES_1_001', 'CA_1', 2.0],
        [4, 'HOBBIES_1_001', 'CA_2', 0.5],
        [4, 'HOBBIES_1_001', 'TX_1', 2.1],
        [4, 'HOBBIES_1_001', 'TX_2', 1.1],
        [4, 'HOBBIES_1_002', 'CA_1', 0.1],
        [4, 'HOBBIES_1_002', 'CA_2', 0.2],
        [4, 'HOBBIES_1_002', 'TX_1', 2.2],
    ]
    spark = SparkSession.builder \
        .appName("tryout") \
        .getOrCreate()
    schema = t.StructType([
        t.StructField('dn', t.IntegerType(), True),
        t.StructField('item_id', t.StringType(), True),
        t.StructField('store_id', t.StringType(), True),
        t.StructField('prediction', t.DoubleType(), True),
    ])

    df = spark.createDataFrame(testval, schema=schema)
    dfp = df.groupBy('item_id', 'store_id').pivot("dn").sum('prediction')

    def rename_col(cn: str) -> str:
        if cn in ['item_id', 'store_id']:
            return cn
        else:
            return f"F{cn}"

    new_cols = [rename_col(c) for c in dfp.columns]

    dfp1 = dfp.toDF(*new_cols)
    dfp1.show()


submission()
