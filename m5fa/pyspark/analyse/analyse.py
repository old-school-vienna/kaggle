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


def analyse01(spark: SparkSession, pppath: Path, pdatdir: Path):
    def key_top(r: t.Row) -> Tuple:
        k = t.Row(year=r['year'], month=r['month'], wday=r['wday'], store_id=r['store_id'], snap=r['snap'],
                  flag_ram=r['flag_ram'], dept_id=r['dept_id'])
        return k, r['label']

    def key_wday(top: Tuple) -> Tuple:
        topkey = top[0]
        k = f"{topkey.wday:02d}"
        return k, top[1]

    def key_dept_id(top: Tuple) -> Tuple:
        topkey = top[0]
        k = f"{topkey.dept_id}"
        return k, top[1]

    def key_dept_id_snap(top: Tuple) -> Tuple:
        topkey = top[0]
        k = f"{topkey.dept_id} {topkey.snap}"
        return k, top[1]

    def key_dept_id_flag_ram(top: Tuple) -> Tuple:
        topkey = top[0]
        k = f"{topkey.dept_id} {topkey.flag_ram}"
        return k, top[1]

    def key_store_id(top: Tuple) -> Tuple:
        topkey = top[0]
        k = f"{topkey.store_id}"
        return k, top[1]

    def key_store_id_snap(top: Tuple) -> Tuple:
        topkey = top[0]
        k = f"{topkey.store_id} {topkey.snap}"
        return k, top[1]

    def key_store_id_flag_ram(top: Tuple) -> Tuple:
        topkey = top[0]
        k = f"{topkey.store_id} {topkey.flag_ram}"
        return k, top[1]

    def key_year(top: Tuple) -> Tuple:
        topkey = top[0]
        k = f"{topkey.year:04d}"
        return k, top[1]

    def key_month(top: Tuple) -> Tuple:
        topkey = top[0]
        k = f"{topkey.month:02d}"
        return k, top[1]

    def plot(top: RDD, name: str, desc: str, fkey: Callable[[Tuple], Tuple], xrot=0):
        ts: RDD = top \
            .map(fkey) \
            .groupByKey() \
            .sortBy(lambda tu: tu[0])

        pbase = Path("/opt/data")
        pplot = pbase / "plot"
        if not pplot.exists():
            pplot.mkdir()

        fig: Figure = plt.figure()
        fig.set_tight_layout("True")
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_title(desc)
        labs = ts.map(lambda tu: tu[0]).collect()
        vals = ts.map(lambda tu: tu[1]).map(list).collect()
        ax.boxplot(vals, labels=labs)
        fig.autofmt_xdate()

        pf1 = pplot / f"box_{name}.png"
        fig.autofmt_xdate(rotation=xrot)
        fig.savefig(pf1)
        print(f"wrote to {pf1}")

    print("--- analyse01 -----------------------")
    print(f"--- Reading: '{pppath}'")
    df: DataFrame = spark.read \
        .parquet(str(pppath))

    ptop = pdatdir / "s5_01_top.parquet"
    if not ptop.exists():
        rddtop: RDD = df.rdd \
            .filter(lambda r: r["label"] is not None) \
            .map(key_top) \
            .reduceByKey(add)
        print(f"--- Writing: '{ptop}'")
        rddtop.toDF().write \
            .format("parquet") \
            .mode("overwrite") \
            .save(str(ptop))
    else:
        print(f"--- Reading: '{ptop}'")
        rddtop = spark.read.parquet(str(ptop)).rdd

    plot(rddtop, "all_months", "Sales for all months", key_month)
    plot(rddtop, "all_years", "Sales for all years", key_year)
    plot(rddtop, "all_stores_snap", "Sales for all stores by snap", key_store_id_snap, xrot=45)
    plot(rddtop, "all_stores_ram", "Sales for all stores by ramadan", key_store_id_flag_ram, xrot=45)
    plot(rddtop, "all_stores", "Sales for all stores", key_store_id)
    plot(rddtop, "all_wday", "Sales for all weekdays", key_wday)
    plot(rddtop, "all_dept", "Sales for all departments", key_dept_id, xrot=45)
    plot(rddtop, "all_dept_snap", "Sales for all departments by snap", key_dept_id_snap, xrot=45)
    plot(rddtop, "all_dept_ram", "Sales for all departments by ramadan flag", key_dept_id_flag_ram, xrot=45)


def run(spark: SparkSession):
    datadir: Path = Path(os.getenv("DATADIR"))
    if datadir is None:
        raise ValueError("Environment variable DATADIR must be defined")
    print(f"datadir = '{datadir}'")

    ppnam = "s5_01"
    pppath = datadir / f"{ppnam}.parquet"
    if not pppath.exists():
        preprocessing(spark, pppath, datadir)
    analyse01(spark, pppath, datadir)


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
