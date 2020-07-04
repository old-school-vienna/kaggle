import datetime
import os
from pprint import pprint

import numpy as np
import pandas as pd
import pyspark.sql as psql
import pyspark.sql.functions as F
from pyspark import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql import SparkSession, DataFrame

import configuration as cfg
import helpers as hlp


def read_parquet_folder_as_pandas(path):
    files = [f for f in os.listdir(path) if f.endswith("parquet")]

    print(f"{len(files)} parquet files found. Beginning reading...")
    start = datetime.datetime.now()

    df_list = [pd.read_parquet(os.path.join(path, f), engine='fastparquet') for f in files]
    df_out = pd.concat(df_list, ignore_index=True)

    end = datetime.datetime.now()
    print(f"Finished. Took {end - start}")
    return df_out


def read_parquet():
    spark = SparkSession.builder \
        .appName(os.path.basename("tryout")) \
        .getOrCreate()
    df: pd.DataFrame = hlp.readFromDatadirParquet(spark, 'sp5_02').toPandas()
    print(np.sort(df.keys().values))
    print(df.shape)


def concat_lists():
    l1 = [0, 1, 2]
    l2 = [4, 5, 6]
    l3 = (l1 + l2)
    pprint(l3)


def extract_ids():
    spark = psql.SparkSession.builder \
        .appName(os.path.basename("tryout")) \
        .getOrCreate()

    list = hlp.read_m5_csv(spark) \
        .groupBy('item_id') \
        .count() \
        .orderBy('item_id') \
        .select('item_id') \
        .collect()

    strlist = [r['item_id'] for r in list]

    pprint(len(strlist))
    pprint(strlist[800:1850])


def check_subm():
    subm_nam = 'GLM_Final_2015.csv'
    datdir = hlp.get_datadir()
    print(f"-- reading {subm_nam}")
    df: pd.DataFrame = pd.read_csv(datdir / subm_nam)
    print(df.shape)
    print(df.keys())
    print(df)
    tmp_nam = datdir / 'tmp.csv'
    print(f"-- writing {tmp_nam}")
    df.to_csv(str(tmp_nam))


def analyse_m5():
    spark = SparkSession.builder \
        .appName("submission") \
        .getOrCreate()
    df = hlp.read_m5_csv(spark)
    df_train = df.where("sales is not null")
    df_prop = df.where("sales is null")
    df_train.describe().show()
    df_prop.describe().show()


def analyse_m5():
    sp = SparkSession.builder \
        .appName("submission") \
        .getOrCreate()
    cols = ["HOBBIES_1_043_CA_3", "HOBBIES_1_043_CA_4", "HOBBIES_1_043_TX_1", "HOBBIES_1_043_TX_2",
            "HOBBIES_1_043_TX_3", "HOBBIES_1_043_WI_1", "HOBBIES_1_043_WI_2", "HOBBIES_1_043_WI_3",
            "dn", "flag_ram", "month", "snap", "vdept_id_0", "vwday_0", "vwday_1",
            "vwday_2", "vwday_3", "vwday_4", "vwday_5", "year"]

    subset_id = 2
    subs_nam, subs_items = cfg.subsets[subset_id]
    fnam = cfg.create_fnam(subs_nam)
    df = hlp.readFromDatadirParquet(sp, fnam) \
        .select(*cols) \
        .orderBy('dn')

    df_1 = df.where("dn between 1900 and 1919")
    df_2 = df.where("dn between 1920 and 1929")
    df_1.show()
    df_2.show()


def analyse_preprop():
    sp = SparkSession.builder \
        .appName("submission") \
        .config("spark.sql.pivotMaxValues", 100000) \
        .getOrCreate()

    fvars = ['year', 'month', 'dn', 'wday', 'snap', 'flag_ram']

    subs_nam, subs_ids = cfg.subsets[0]

    df01 = hlp.read_m5_csv(sp) \
        .where(F.col('item_id').isin(*subs_ids)) \
        .withColumn('subm_id', F.concat(F.col('item_id'), F.lit('_'), F.col('store_id'))) \
        .drop('item_id', 'store_id', 'Sales_Pred') \
        .groupBy(*fvars) \
        .pivot('subm_id') \
        .agg(F.sum('sales')) \
        .na.fill(0.0) \
        .orderBy('dn')

    df01.show()
    print("-------- A FINISHED ------------------------------------------------------------")

    stages = []
    catvars = ['wday']

    for v in catvars:
        stages += [StringIndexer(inputCol=v,
                                 outputCol=f"i{v}")]
    stages += [OneHotEncoder(inputCols=[f"i{v}" for v in catvars],
                             outputCols=[f"v{v}" for v in catvars])]

    pip: Pipeline = Pipeline(stages=stages)
    pipm = pip.fit(df01)
    df01: DataFrame = pipm.transform(df01)
    catvarsi = [f"i{n}" for n in catvars]
    ppdf = df01.drop(*(catvarsi + catvars))

    rdd1 = ppdf.rdd.map(hlp.one_hot_row)

    ctx: SQLContext = SQLContext.getOrCreate(sp.sparkContext)
    df1 = ctx.createDataFrame(rdd1)

    df1.show()
    print("-------- B FINISHED ------------------------------------------------------------")

    rdd1 = ppdf.rdd.map(hlp.one_hot_row)

    ctx: SQLContext = SQLContext.getOrCreate(sp.sparkContext)
    df1 = ctx.createDataFrame(rdd1)
    df1.show()
    print("-------- C FINISHED ------------------------------------------------------------")


analyse_preprop()
