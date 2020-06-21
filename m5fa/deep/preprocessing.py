import os

import pyspark.sql.types as T
from pyspark import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql import DataFrame, SparkSession

import helpers as hlp


def preprocessing():
    print("--- preprocessing -----------------------")

    spark = SparkSession.builder \
        .appName(os.path.basename("preporcessing")) \
        .config("spark.driver.memory", "25g") \
        .getOrCreate()

    nam = 'sp5_02'
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

    sales5: DataFrame = spark.read.csv(str(csv_path), header='true', schema=schema)

    stages = []
    catvars = ['dept_id', 'item_id', 'store_id', 'wday']
    for v in catvars:
        stages += [StringIndexer(inputCol=v,
                                 outputCol=f"i{v}")]
    stages += [OneHotEncoder(inputCols=[f"i{v}" for v in catvars],
                             outputCols=[f"v{v}" for v in catvars])]

    pip: Pipeline = Pipeline(stages=stages)
    pipm = pip.fit(sales5)
    df: DataFrame = pipm.transform(sales5)
    ppdf = df.drop('idept_id', 'iitem_id', 'istore_id', 'iwday')

    rdd1 = ppdf.rdd.map(hlp.one_hot_row)

    ctx: SQLContext = SQLContext.getOrCreate(spark.sparkContext)
    df1 = ctx.createDataFrame(rdd1)
    df1.show()
    print(f"--- Writing: '{nam}'")
    hlp.writeToDatadirParquet(df1, nam)

    spark.stop()


if __name__ == '__main__':
    preprocessing()
