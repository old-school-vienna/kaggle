from dataclasses import dataclass

from pyspark.ml import Transformer
from pyspark.sql import SparkSession, DataFrame

import common as cm
import helpers as hlp

import pyspark.sql.functions as F
import pyspark.sql.types as T


@dataclass
class Model:
    desc: str
    id: str


models = [
    Model("glr poisson identity", "glrpi"),
    Model("glr gaussian identity", "glrgi"),
    Model("glr poisson log", "glrpl"),
]


def load_model(model: Model) -> Transformer:
    fnam = cm.fnam(model.id)
    return hlp.load_model(hlp.get_datadir(), fnam)


def rename_cols(df: DataFrame) -> DataFrame:
    def rename_col(cn: str) -> str:
        if cn in ['id']:
            return cn
        else:
            return f"F{cn}"

    new_cols = [rename_col(c) for c in df.columns]
    return df.toDF(*new_cols)


spark = SparkSession.builder \
    .appName("submission") \
    .getOrCreate()

for m in models:
    print(f"-- process {m}")
    t: Transformer = load_model(m)
    pmap = t.extractParamMap()
    print(f"-- Loaded {m}")

    test_df = hlp.readFromDatadirParquet(spark, "s5_01") \
        .where(F.col("label").isNull())

    pred = t.transform(test_df)

    vali_df = pred \
        .where(pred.dn <= 1941) \
        .withColumn("dn1", pred.dn - 1913) \
        .withColumn("id", F.concat(F.col('item_id'), F.lit('_'), F.col('store_id'), F.lit('_validation'))) \
        .withColumn("ipred", F.col('prediction').cast(T.IntegerType())) \
        .groupBy('id') \
        .pivot("dn1") \
        .sum('ipred')

    evalu_df = pred \
        .where(pred.dn > 1941) \
        .withColumn("dn1", pred.dn - 1941) \
        .withColumn("id", F.concat(F.col('item_id'), F.lit('_'), F.col('store_id'), F.lit('_evaluation'))) \
        .withColumn("ipred", F.col('prediction').cast(T.IntegerType())) \
        .groupBy('id') \
        .pivot("dn1") \
        .sum('ipred')

    vali_pdf = rename_cols(vali_df) \
        .orderBy('id')
    vali_pdf.show()

    evalu_pdf = rename_cols(evalu_df) \
        .orderBy('id')
    evalu_pdf.describe().show()

    submission = vali_pdf.union(evalu_pdf)
    spath = hlp.get_datadir() / f'subm_{m.id}.csv'
    submission.toPandas().to_csv(str(spath), header=True, index=False)
    print("-----------------------------------------------------")
    print(f"-- Wrote submission for {m.desc} to {spath.absolute()}")
    print()
    print()
