import os
from pathlib import Path

import pyspark.sql.types as st
from pyspark.sql.types import Row
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame, SparkSession

spark = SparkSession.builder \
    .appName("karl02") \
    .getOrCreate()

datadir: str = os.getenv("DATADIR")
if datadir is None:
    raise ValueError("Environment variable DATADIR must be defined")
print(f"datadir = '{datadir}'")

schema = st.StructType([
    st.StructField('year', st.IntegerType(), True),
    st.StructField('month', st.IntegerType(), True),
    st.StructField('dn', st.IntegerType(), True),
    st.StructField('wday', st.IntegerType(), True),
    st.StructField('snap', st.IntegerType(), True),
    st.StructField('dept_id', st.StringType(), True),
    st.StructField('item_id', st.StringType(), True),
    st.StructField('store_id', st.StringType(), True),
    st.StructField('sales', st.DoubleType(), True),
    st.StructField('flag_ram', st.IntegerType(), True),
    st.StructField('Sales_Pred', st.DoubleType(), True)
])

p = str(Path(datadir, "Sales5_Ab2011_InklPred.csv"))
print(f"Reading: '{p}'")

train: DataFrame = spark.read.csv(p, header='true', schema=schema)
rows = train.rdd.take(5)
for r in rows:
    dn = r["sales"]
    d = r.asDict()
    v = list(d.values())
    print(v)
    print(type(v))

print("------------------------- R E A D Y --------------------------------")


def train(df: DataFrame):
    def astraining(row: Row) -> Row:
        df = row.asDict()
        del df['Sales_Pred']
        del df['sales']
        sales = row.asDict()['sales']
        return Row(label=sales, features=list(df.values()))

    t3 = train.rdd \
        .filter(lambda r: r["sales"] is not None) \
        .map(astraining)

    gbt = GBTRegressor(maxIter=10)
    df = spark.createDataFrame(t3)
    df.show()
    gbt.fit(df)
    print("----------- after fit ------------")
