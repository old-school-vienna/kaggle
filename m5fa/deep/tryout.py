import datetime
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pprint import pprint
import pyspark.sql as psql

import helpers as hlp
import numpy as np


def get_datadir() -> Path:
    env = os.getenv("DATADIR")
    if env is None:
        raise ValueError("Environment variable DATADIR must be defined")
    dd = Path(env)
    if not dd.exists():
        raise ValueError(f"Environment variable DATADIR must define an existing directory. {dd}")
    return dd


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


def read_csv():
    p = Path(get_datadir()) / 'Sales5_Ab2011_InklPred.csv'
    print(f"reading {p}")
    df: pd.DataFrame = pd.read_csv(str(p))
    print(df.shape)
    print(df.keys())

    cat_vars = ['year', 'month', 'wday', 'snap', 'dept_id', 'item_id', 'store_id', 'flag_ram']
    cont_vars = ['dn', 'sales', 'Sales_Pred']

    df1 = None
    for cv in cat_vars:
        du = pd.get_dummies(df[cv], prefix=cv)
        if df1 is None:
            df1 = du
        else:
            df1 = df1.join(du)

    for cv in cont_vars:
        df1 = df1.join(df[cv])

    df1 = df1.sort_values(by=['dn'])

    print(len(df1.keys()))
    print(df1.keys())
    print(df1)


def concat_lists():
    l1 = [0, 1, 2]
    l2 = [4, 5, 6]
    l3 = (l1 + l2)
    pprint(l3)


def extract_ids():
    spark = psql.SparkSession.builder \
        .appName(os.path.basename("tryout")) \
        .getOrCreate()

    list = hlp.read_csv(spark, 'Sales5_Ab2011_InklPred.csv') \
        .groupBy('item_id') \
        .count() \
        .orderBy('item_id') \
        .select('item_id') \
        .collect()

    strlist = [r['item_id'] for r in list]

    pprint(len(strlist))
    pprint(strlist[800:1850])


