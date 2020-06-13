import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyspark.sql.functions as sf
from pyspark.ml.util import MLWritable, MLReadable
from pyspark.sql import DataFrame, SparkSession

import importlib


@dataclass
class Classnam:
    module: str
    nam: str


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
        .where(sf.col("label").isNotNull()) \
        .sample(train_size / big_size)
    test = big \
        .where(sf.col("label").isNull()) \
        .sample(test_size / big_size)
    small = train.union(test)
    writeToDatadirParquet(small, dest_nam)


def classnam_from_filenam(fnam: str) -> Classnam:
    snam = fnam.split("__")
    return Classnam(snam[0], snam[1])


def create_nam_from_classname(cn: Classnam, nam: str) -> str:
    return f"{cn.module}__{cn.nam}__{nam}"


def classnam_from_obj(obj: Any) -> Classnam:
    mod = obj.__module__
    cn = obj.__class__.__name__
    return Classnam(mod, cn)


def save_model(model: MLWritable, bdir: Path, nam: str) -> str:
    cn = classnam_from_obj(model)
    nams = [f.name for f in bdir.iterdir()]
    if file_for_other_class_exists(nams, cn, nam):
        raise ValueError(f"File for name {nam} exists for other class")

    cn = classnam_from_obj(model)
    cnam = create_nam_from_classname(cn, nam)
    model.write().overwrite().save(str(bdir / cnam))
    return cnam


def file_for_other_class_exists(fnams: Iterable[str], cn: Classnam, nam: str):
    cns = [classnam_from_filenam(fnam) != cn for fnam in fnams if fnam.endswith(nam)]
    return any(cns)


def load_model(bdir: Path, nam: str) -> Any:
    def find_dedicated_file() -> Path:
        matching = [p for p in bdir.iterdir() if p.name.endswith(nam)]
        if len(matching) == 0:
            raise ValueError(f"Found no file for {nam} in {bdir}")
        if len(matching) > 1:
            sm = [str(p.name) for p in matching]
            s = ', '.join(sm)
            raise ValueError(f"Found more than one file for {nam} in {bdir}. [{s}]")
        return matching[0]

    path = find_dedicated_file()
    cn = classnam_from_filenam(path.name)

    module = importlib.import_module(cn.module)
    my_class = getattr(module, cn.nam)
    c: MLReadable = my_class()
    return c.load(str(path))


if __name__ == "__main__":
    spark = SparkSession.builder\
        .appName("helpers")\
        .getOrCreate()
    create_small_dataframe(spark)
