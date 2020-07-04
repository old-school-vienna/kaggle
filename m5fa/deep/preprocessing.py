import os
from pyspark import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

import helpers as hlp
import configuration as cfg


def read(sp: SparkSession, items: list) -> DataFrame:
    print("--- read -----------------------")
    """
        The where clause for small datasets    
        .where(F.col('item_id').isin("FOODS_1_001", "HOBBIES_1_021", "HOUSEHOLD_2_491")) \
    """
    fvars = ['year', 'month', 'dn', 'wday', 'snap', 'flag_ram']
    return hlp.read_m5_csv(sp) \
        .where(F.col('item_id').isin(*items)) \
        .withColumn('subm_id', F.concat(F.col('item_id'), F.lit('_'), F.col('store_id'))) \
        .drop('item_id', 'store_id', 'Sales_Pred') \
        .groupBy(*fvars) \
        .pivot('subm_id') \
        .agg(F.sum('sales')) \
        .na.fill(0.0) \
        .orderBy('dn')


def preprocessing(sp: SparkSession, subset_id: int):
    # Select here one of the predefined subsets
    subs_name, subs_items = cfg.subsets[subset_id]

    fnam = cfg.create_fnam(subs_name)
    print(f"--- preprocessing --- output :{fnam}--------------------")

    df01: DataFrame = read(sp, subs_items)

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
    print(f"--- Writing: '{fnam}'")
    hlp.writeToDatadirParquet(df1, fnam)


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName(os.path.basename("preporcessing")) \
        .config("spark.sql.pivotMaxValues", 100000) \
        .getOrCreate()
    subset_id = 2
    preprocessing(spark, subset_id)
