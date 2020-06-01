import os
import time
from pathlib import Path

import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import DataFrame, SparkSession
from pyspark import Row


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
    df: DataFrame = pipm.transform(s5)
    return df.drop('idept_id', 'iitem_id', 'istore_id', 'iwday', 'vdept_id', 'vtem_id', 'vstore_id', 'vwday')


def nulls(row: Row) -> Row:
    d = row.asDict()
    _cnt = 0
    for _var in d.keys():
        if d[_var] is None:
            _cnt += 1
    d['nullcnt'] = _cnt
    return Row(**d)


def preprocessing(spark: SparkSession, pppath: Path, datadir: str):
    print("--- preprocessing -----------------------")

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

    ppdf = prepro(sales5)
    print(f"--- Writing: '{pppath}'")
    ppdf.write \
        .format("parquet") \
        .mode("overwrite") \
        .save(str(pppath))


def process(spark: SparkSession, pppath: Path, datadir: str):
    print("--- process -----------------------")
    print(f"--- Reading: '{pppath}'")
    df: DataFrame = spark.read \
        .parquet(str(pppath))

    train = df \
        .where(F.col("label").isNotNull())
    test = df \
        .where(F.col("label").isNull())

    lr = LinearRegression()
    model = lr.fit(train)

    pred = model.transform(test)
    pred.show()


def run():
    spark = SparkSession.builder \
        .appName("karl04") \
        .getOrCreate()

    datadir: str = os.getenv("DATADIR")
    if datadir is None:
        raise ValueError("Environment variable DATADIR must be defined")
    print(f"datadir = '{datadir}'")

    ppnam = "s5_01"
    pppath = Path(datadir, f"{ppnam}.parquet")
    if not pppath.exists():
        preprocessing(spark, pppath, datadir)
    process(spark, pppath, datadir)


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
