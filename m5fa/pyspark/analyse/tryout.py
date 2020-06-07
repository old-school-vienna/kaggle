from dataclasses import dataclass
from pprint import pprint

from pyspark import SparkContext
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql import SparkSession
import pyspark.sql.types as t


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
    dfp.show()


submission()
