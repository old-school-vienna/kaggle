import os
from pathlib import Path
from typing import Any

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, SparkSession


def read(spark: SparkSession, datadir: Path, nam: str) -> DataFrame:
    print(f"datadir = '{datadir}'")

    path = datadir / f"{nam}.parquet"
    return spark.read.parquet(str(path))


def write(df: DataFrame, datadir: Path, nam: str):
    small_path = datadir / f"{nam}.parquet"
    print(f"--- Writing: '{small_path}'")
    df.write \
        .format("parquet") \
        .mode("overwrite") \
        .save(str(small_path))


def get_datadir() -> Path:
    dd: Path = Path(os.getenv("DATADIR"))
    if dd is None:
        raise ValueError("Environment variable DATADIR must be defined")
    return dd


def create_small_dataframe():
    print("creating small dataset")
    sps = SparkSession.builder \
        .appName(os.path.basename(__file__)) \
        .getOrCreate()
    dd = get_datadir()
    big: DataFrame = read(sps, dd, "s5_01")
    train = big \
        .where(sf.col("label").isNotNull()) \
        .limit(200)
    test = big \
        .where(sf.col("label").isNull()) \
        .limit(50)
    small = train.union(test)
    write(small, dd, "s5_01_small")


def fval(value: Any, leng: int) -> str:
    if value is None:
        fstr = f"{{:{leng}}}"
        return fstr.format('None')
    if isinstance(value, int):
        fstr = f"{{:{leng}d}}"
        return fstr.format(value)
    if isinstance(value, float):
        fstr = f"{{:{leng}.3f}}"
        return fstr.format(value)
    else:
        fstr = f"{{:{leng}}}"
        return fstr.format(str(value))


if __name__ == "__main__":
    create_small_dataframe()
