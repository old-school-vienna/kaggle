import os
from pathlib import Path

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import RDD
from pyspark.sql import DataFrame, SparkSession

spark = SparkSession.builder \
    .appName("karl01") \
    .getOrCreate()

datadir: str = os.getenv("DATADIR")
if datadir is None:
    raise ValueError("Environment variable DATADIR must be defined")
print(f"datadir = '{datadir}'")

schema = T.StructType([
    T.StructField('year', T.IntegerType(), True),
    T.StructField('month', T.IntegerType(), True),
    T.StructField('dn', T.IntegerType(), True),
    T.StructField('wday', T.IntegerType(), True),
    T.StructField('snap', T.IntegerType(), True),
    T.StructField('dept_id', T.StringType(), True),
    T.StructField('item_id', T.StringType(), True),
    T.StructField('store_id', T.StringType(), True),
    T.StructField('sales', T.DoubleType(), True),
    T.StructField('flag_ram', T.IntegerType(), True),
    T.StructField('Sales_Pred', T.DoubleType(), True)
])


def mse(row: T.Row) -> T.Row:
    d = row.asDict()
    _mse = 0.0
    if d['Sales_Pred'] is None:
        print("'Sales_Pred'=None")
        _mse = 0
    elif d['sales'] is None:
        _mse = d['Sales_Pred'] ** 2
    else:
        _mse = (d['Sales_Pred'] - d['sales']) ** 2
    d['mse'] = _mse
    return T.Row(**d)


def keyvalues(row: T.Row) -> ((str, str), float):
    d = row.asDict()
    key = (d["store_id"], d["dept_id"], d["year"])
    return key, d["mse"]


p = str(Path(datadir, "Sales5_Ab2011_InklPred.csv"))
print(f"Reading: '{p}'")

train: DataFrame = spark.read.csv(p, header='true', schema=schema)
t: RDD = train.rdd
t1 = t.map(mse)
t1.persist()
t2 = t1 \
    .map(keyvalues) \
    .reduceByKey(lambda a, x: a + x)

t2.persist()
t3 = t2.collect()

for r in t3:
    print(r)


def df(train: DataFrame):
    """
    Some experiments
    """
    cols = train.columns
    print(cols)
    t1 = train.withColumn("mse", F.pow((F.col("Sales_Pred") - F.col("sales")), 2))
    t1.show()
