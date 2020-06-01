import os
import time
from pathlib import Path

import pyspark.sql.types as t
import pyspark.sql.functions as f
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.sql import DataFrame, SparkSession


def preprocessing(spark: SparkSession, pppath: Path, datadir: str):
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

    csv_path = str(Path(datadir, "Sales5_Ab2011_InklPred.csv"))
    print(f"--- Reading: '{csv_path}'")

    sales5: DataFrame = spark.read.csv(csv_path, header='true', schema=schema) \
        .withColumn("label", f.col('sales'))

    ppdf = prepro(sales5)
    print(f"--- Writing: '{pppath}'")
    ppdf.write \
        .format("parquet") \
        .mode("overwrite") \
        .save(str(pppath))


def analyse01(spark: SparkSession, pppath: Path):
    print("--- analyse01 -----------------------")
    print(f"--- Reading: '{pppath}'")
    df: DataFrame = spark.read \
        .parquet(str(pppath))

    train = df \
        .where(f.col("label").isNotNull())

    train.show()


def run(spark: SparkSession):
    datadir: str = os.getenv("DATADIR")
    if datadir is None:
        raise ValueError("Environment variable DATADIR must be defined")
    print(f"datadir = '{datadir}'")

    ppnam = "s5_01"
    pppath = Path(datadir, f"{ppnam}.parquet")
    if not pppath.exists():
        preprocessing(spark, pppath, datadir)
    analyse01(spark, pppath)


def main():
    start = time.time()
    spark = SparkSession.builder \
        .appName("karl04") \
        .getOrCreate()
    run(spark)
    end = time.time()
    elapsed = end - start
    elapseds = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"------------------------- R E A D Y ------------ {elapseds} --------------------")
    input("Press Enter to continue...")
    spark.stop()


if __name__ == '__main__':
    main()
