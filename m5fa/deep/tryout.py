import datetime
import os
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import pyspark.sql as psql
from pyspark.sql import SparkSession

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
