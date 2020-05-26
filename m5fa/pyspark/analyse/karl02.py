import os
from pathlib import Path

import pyspark.sql.types as T
from pyspark import Row
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame, SparkSession

spark = SparkSession.builder \
    .appName("karl02") \
    .config("spark.driver.memory", "12g") \
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


def astraining(row: Row) -> Row:
    df = row.asDict()
    del df['Sales_Pred']
    del df['sales']
    sales = row.asDict()['sales']
    return Row(label=sales, features=list(df.values()))


p = str(Path(datadir, "Sales5_Ab2011_InklPred.csv"))
print(f"Reading: '{p}'")

train: DataFrame = spark.read.csv(p, header='true', schema=schema)
t3 = train.rdd \
    .filter(lambda r: r["sales"] is not None) \
    .map(astraining)

gbt = GBTRegressor(maxIter=10)
df = spark.createDataFrame(t3)
df.show()
gbt.fit(df)
print("------------------------- R E A D Y --------------------------------")
