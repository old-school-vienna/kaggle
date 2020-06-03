import os
import time
from pathlib import Path
from typing import Tuple, Callable

import matplotlib.pyplot as plt
import pyspark.sql.types as t
import pyspark.sql.functions as f
from operator import add
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyspark import RDD
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.sql import DataFrame, SparkSession


def preprocessing(spark: SparkSession, pppath: Path, datadir: Path):
    def prepro(s5: DataFrame) -> DataFrame:
        stages = []
        catvars = ['dept_id', 'item_id', 'store_id', 'wday']
        for v in catvars:
            stages += [StringIndexer(inputCol=v,
                                     outputCol=f"i{v}")]
        stages += [OneHotEncoderEstimator(inputCols=[f"i{v}" for v in catvars],
                                          outputCols=[f"v{v}" for v in catvars])]
        stages += [VectorAssembler(inputCols=['vwday', 'vitem_id', 'vdept_id', 'vstore_id', 'flag_ram',
                                              'snap', 'dn', 'month', 'year'],
                                   outputCol='features')]

        pip: Pipeline = Pipeline(stages=stages)
        pipm = pip.fit(s5)
        df: DataFrame = pipm.transform(s5)
        return df.drop('idept_id', 'iitem_id', 'istore_id', 'iwday', 'vdept_id', 'vtem_id', 'vstore_id', 'vwday')

    print("--- preprocessing -----------------------")

    schema = t.StructType([
        t.StructField('year', t.IntegerType(), True),
        t.StructField('month', t.IntegerType(), True),
        t.StructField('dn', t.IntegerType(), True),
        t.StructField('wday', t.IntegerType(), True),
        t.StructField('snap', t.IntegerType(), True),
        t.StructField('dept_id', t.StringType(), True),
        t.StructField('item_id', t.StringType(), True),
        t.StructField('store_id', t.StringType(), True),
        t.StructField('sales', t.DoubleType(), True),
        t.StructField('flag_ram', t.IntegerType(), True),
        t.StructField('Sales_Pred', t.DoubleType(), True)
    ])

    csv_path = datadir / "Sales5_Ab2011_InklPred.csv"
    print(f"--- Reading: '{csv_path}'")

    sales5: DataFrame = spark.read.csv(str(csv_path), header='true', schema=schema) \
        .withColumn("label", f.col('sales'))

    ppdf = prepro(sales5)
    print(f"--- Writing: '{pppath}'")
    ppdf.write \
        .format("parquet") \
        .mode("overwrite") \
        .save(str(pppath))


def pipeline(spark: SparkSession, pppath: Path, datadir: Path):
    print(f"--- Reading: '{pppath}'")

    pp: DataFrame = spark.read.parquet(str(pppath))
    pp.show()


def run(spark: SparkSession):
    datadir: Path = Path(os.getenv("DATADIR"))
    if datadir is None:
        raise ValueError("Environment variable DATADIR must be defined")
    print(f"datadir = '{datadir}'")

    ppnam = "s5_01"
    pppath = datadir / f"{ppnam}.parquet"
    if not pppath.exists():
        preprocessing(spark, pppath, datadir)
    pipeline(spark, pppath, datadir)


def main():
    start = time.time()
    spark = SparkSession.builder \
        .appName("analyse") \
        .getOrCreate()
    run(spark)
    end = time.time()
    elapsed = end - start
    elapseds = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"------------------------- R E A D Y ------------ {elapseds} --------------------")
    spark.stop()
    exit(0)


if __name__ == '__main__':
    main()
