import os
import time
from pathlib import Path

import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame, SparkSession


def nullrows_cnt(df: DataFrame) -> int:
    return df.rdd \
        .map(nulls) \
        .filter(lambda row: row['nullcnt'] > 0) \
        .count()


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
    dft: DataFrame = pipm.transform(s5)
    return dft.drop('idept_id', 'iitem_id', 'istore_id', 'iwday', 'vdept_id', 'vtem_id', 'vstore_id', 'vwday')


def process(sales5: DataFrame):
    df = prepro(sales5)
    train = df \
        .where(F.col("label").isNotNull())
    test = df \
        .where(F.col("label").isNull())

    lr = LinearRegression()
    model = lr.fit(train)

    pred = model.transform(test)
    pred.show()


def nulls(row: T.Row) -> T.Row:
    d = row.asDict()
    _cnt = 0
    for _var in d.keys():
        if d[_var] is None:
            _cnt += 1
    d['nullcnt'] = _cnt
    return T.Row(**d)


def run():
    spark = SparkSession.builder \
        .appName("karl03") \
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

    csv_path = str(Path(datadir, "Sales5_Ab2011_InklPred.csv"))
    print(f"--- Reading: '{csv_path}'")

    sales5: DataFrame = spark.read.csv(csv_path, header='true', schema=schema) \
        .withColumn("label", F.col('sales'))

    orig_path = str(Path(datadir, "Sales5_orig.parquet"))
    print(f"--- Writing: '{orig_path}'")
    sales5.write \
        .format("parquet") \
        .mode("overwrite") \
        .save(orig_path)

    print(f"--- Reading: '{orig_path}'")
    df1: DataFrame = spark.read \
        .parquet(orig_path)

    df1.show()


def main():
    start = time.time()
    run()
    end = time.time()
    elapsed = end - start
    elapseds = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"------------------------- R E A D Y ------------ {elapseds} --------------------")
    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
