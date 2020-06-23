import os
from pathlib import Path
from typing import Any

import pyspark.sql.functions as F
from pyspark import Row
from pyspark.sql import DataFrame, SparkSession


def readFromDatadirParquet(spark: SparkSession, nam: str) -> DataFrame:
    path = get_datadir() / f"{nam}.parquet"
    return spark.read.parquet(str(path))


def writeToDatadirParquet(df: DataFrame, nam: str):
    small_path = get_datadir() / f"{nam}.parquet"
    print(f"--- Writing: '{small_path}'")
    df.write \
        .format("parquet") \
        .mode("overwrite") \
        .save(str(small_path))


def get_datadir() -> Path:
    env = os.getenv("DATADIR")
    if env is None:
        raise ValueError("Environment variable DATADIR must be defined")
    dd = Path(env)
    if not dd.exists():
        raise ValueError(f"Environment variable DATADIR must define an existing directory. {dd}")
    return dd


def create_small_dataframe(sps: SparkSession):
    _create_small_df('s5_01', 'small', 200, 50, sps)


def create_medium_dataframe(sps: SparkSession):
    _create_small_df('s5_01', 'medium', 2000, 500, sps)


def _create_small_df(base_name: str, qual: str, train_size: int, test_size: int, sps: SparkSession):
    dest_nam = f"{base_name}_{qual}"
    print(f"creating dataset. {base_name} -> {dest_nam}")
    big: DataFrame = readFromDatadirParquet(sps, base_name)
    big_size = float(big.count())
    train = big \
        .where(F.col("label").isNotNull()) \
        .sample(train_size / big_size)
    test = big \
        .where(F.col("label").isNull()) \
        .sample(test_size / big_size)
    small = train.union(test)
    writeToDatadirParquet(small, dest_nam)


def one_hot_row(r: Row) -> Row:
    def c_to_dict(d: dict, k: str, v: Any):
        d.update({k: v})

    def cv_to_dict(d: dict, k: str, v: Any, l: int):
        for i in range(l):
            d.update({f"{k}_{i}": float(v[i])})

    d1 = r.asDict()
    do = {}
    c_to_dict(do, 'year', d1['year'])
    c_to_dict(do, 'month', d1['month'])
    c_to_dict(do, 'dn', d1['dn'])
    c_to_dict(do, 'snap', d1['snap'])
    c_to_dict(do, 'flag_ram', d1['flag_ram'])
    c_to_dict(do, 'sales', d1['sales'])
    c_to_dict(do, 'Sales_Pred', d1['Sales_Pred'])
    cv_to_dict(do, 'dept_id', d1['vdept_id'], 6)
    cv_to_dict(do, 'item_id', d1['vitem_id'], 3048)
    cv_to_dict(do, 'store_id', d1['vstore_id'], 9)
    cv_to_dict(do, 'wday', d1['vwday'], 6)
    return Row(**do)


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("helpers") \
        .getOrCreate()
    create_small_dataframe(spark)
