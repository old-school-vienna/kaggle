import os
import time
from pathlib import Path

import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.sql import DataFrame, SparkSession


def run():
    spark = SparkSession.builder \
        .appName("karl02") \
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

    p = str(Path(datadir, "Sales5_Ab2011_InklPred.csv"))
    print(f"Reading: '{p}'")

    train: DataFrame = spark.read.csv(p, header='true', schema=schema) \
        .withColumn("label", F.col('sales'))
    

    catvars = ['dept_id', 'item_id', 'store_id', 'wday']

    stages = []

    for v in catvars:
        stages += [StringIndexer(inputCol=v,
                                 outputCol=f"i{v}")]
    ohin = [f"i{v}" for v in catvars]
    ohout = [f"v{v}" for v in catvars]
    stages += [OneHotEncoderEstimator(inputCols=ohin,
                                      outputCols=ohout)]
    stages += [VectorAssembler(inputCols=['vwday', 'vitem_id', 'vdept_id', 'vstore_id', 'flag_ram',
                                          'snap', 'dn', 'month', 'year'],
                               outputCol='features')]

    pip = Pipeline(stages=stages)
    pipm = pip.fit(train)

    dft: DataFrame = pipm.transform(train)
    dft.drop('idept_id', 'iitem_id', 'istore_id', 'iwday', 'vdept_id', 'vtem_id', 'vstore_id', 'vwday').show()


start = time.time()
run()
end = time.time()
elapsed = end - start
elapseds = time.strftime("%H:%M:%S", time.gmtime(elapsed))
print(f"------------------------- R E A D Y ------------ {elapseds} --------------------")
