import os
from collections import ChainMap
from pathlib import Path
from typing import Any

import pyspark.sql.functions as F
from pyspark import Row
from pyspark.ml.linalg import Vector
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
    def c_to_dict(k: str, v: Any) -> dict:
        if isinstance(v, Vector):
            return dict([(f"{k}_{i}", float(v[i])) for i in range(len(v))])
        else:
            return {k: v}

    d = r.asDict()
    dicts: list = [c_to_dict(k, d[k]) for k in d.keys()]
    di = dict(ChainMap(*dicts))
    return Row(**di)


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("helpers") \
        .getOrCreate()
    create_small_dataframe(spark)
