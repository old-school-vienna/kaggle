import datetime
import os
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession

import helpers as hlp


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
        .appName(os.path.basename("preporcessing")) \
        .getOrCreate()
    df: pd.DataFrame = hlp.readFromDatadirParquet(spark, 'sp5_02').toPandas()
    print(df.keys())
    print(df)


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


read_parquet()
