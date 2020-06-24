import os
from pprint import pprint

import pyspark.sql.types as T
from pyspark import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

import helpers as hlp


def read_csv(spark: SparkSession) -> DataFrame:
    datadir = hlp.get_datadir()

    schema = T.StructType([
        T.StructField('year', T.IntegerType(), True),
        T.StructField('month', T.IntegerType(), True),
        T.StructField('dn', T.IntegerType(), True),
        T.StructField('wday', T.IntegerType(), True),
        T.StructField('snap', T.IntegerType(), True),
        T.StructField('dept_id', T.StringType(), True),
        T.StructField('item_id', T.StringType(), True),
        T.StructField('store_id', T.StringType(), True),
        T.StructField('sales', T.DoubleType(), True),
        T.StructField('flag_ram', T.IntegerType(), True),
        T.StructField('Sales_Pred', T.DoubleType(), True),
    ])

    csv_path = datadir / "Sales5_Ab2011_InklPred.csv"
    print(f"--- Reading: '{csv_path}'")

    return spark.read.csv(str(csv_path), header='true', schema=schema)


def preprocessing(sp: SparkSession):
    print("--- preprocessing -----------------------")

    nam = 'sp5_02'
    fvars = ['year', 'month', 'dn', 'wday', 'snap', 'dept_id', 'flag_ram']
    catvars = ['dept_id', 'wday']
    df01: DataFrame = read_csv(sp) \
        .where(F.col('item_id').isin("FOODS_1_001", "HOBBIES_1_021", "HOUSEHOLD_2_491")) \
        .withColumn('subm_id', F.concat(F.col('item_id'), F.lit('_'), F.col('store_id'))) \
        .drop('item_id', 'store_id', 'Sales_Pred') \
        .groupBy(*fvars) \
        .pivot('subm_id') \
        .agg(F.sum('sales')) \
        .na.fill(0.0) \
        .orderBy('dn')

    df01.show()
    df01.describe().show()
    pprint(df01.toPandas().dtypes)

    stages = []
    for v in catvars:
        stages += [StringIndexer(inputCol=v,
                                 outputCol=f"i{v}")]
    stages += [OneHotEncoder(inputCols=[f"i{v}" for v in catvars],
                             outputCols=[f"v{v}" for v in catvars])]

    pip: Pipeline = Pipeline(stages=stages)
    pipm = pip.fit(df01)
    df01: DataFrame = pipm.transform(df01)
    catvarsi = [f"i{n}" for n in catvars]
    ppdf = df01.drop(*catvarsi)

    rdd1 = ppdf.rdd.map(hlp.one_hot_row)

    ctx: SQLContext = SQLContext.getOrCreate(sp.sparkContext)
    df1 = ctx.createDataFrame(rdd1)
    df1.show()
    print(f"--- Writing: '{nam}'")
    hlp.writeToDatadirParquet(df1, nam)
    sp.stop()


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName(os.path.basename("preporcessing")) \
        .getOrCreate()

    preprocessing(spark)
