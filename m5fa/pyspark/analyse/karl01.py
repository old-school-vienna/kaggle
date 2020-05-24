import os

from pyspark import Row, RDD
from pyspark.shell import spark
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T

datadir: str = os.getenv("DATADIR")

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


def mse(row: Row) -> Row:
    d = row.asDict()
    sp = d['Sales_Pred']
    s = d['sales']
    d['mse'] = (sp + s) ** 2
    return Row(**d)


train: DataFrame = spark.read.csv(f"{datadir}/Sales5_Ab2011_InklPred.csv", header=True, schema=schema)
t: RDD = train.rdd
f = t \
    .map(mse) \
    .take(5)

for r in f:
    print(r)

def df(train: DataFrame):
    """
    Some experiments
    """
    cols = train.columns
    print(cols)
    t1 = train.withColumn("mse", F.pow((F.col("Sales_Pred") - F.col("sales")), 2))
    t1.show()
