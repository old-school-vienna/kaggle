import os
from pathlib import Path

import pyspark.sql.functions as sfunc
import pyspark.sql.types as stype
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.sql import DataFrame, SparkSession

import helpers as hlp


def prepro(spark: SparkSession, datadir: Path, nam: str):
    def pp(s5: DataFrame) -> DataFrame:
        stages = []
        catvars = ['dept_id', 'item_id', 'store_id', 'wday']
        for v in catvars:
            stages += [StringIndexer(inputCol=v,
                                     outputCol=f"i{v}")]
        stages += [OneHotEncoderEstimator(inputCols=[f"i{v}" for v in catvars],
                                          outputCols=[f"v{v}" for v in catvars])]
        stages += [VectorAssembler(inputCols=['vwday', 'vitem_id', 'vdept_id', 'vstore_id', 'flag_ram',
                                              'snap', 'dn', 'month', 'year'],
                                   outputCol='features')]

        pip: Pipeline = Pipeline(stages=stages)
        pipm = pip.fit(s5)
        df: DataFrame = pipm.transform(s5)
        return df.drop('idept_id', 'iitem_id', 'istore_id', 'iwday', 'vdept_id', 'vtem_id', 'vstore_id', 'vwday')

    print("--- preprocessing -----------------------")

    schema = stype.StructType([
        stype.StructField('year', stype.IntegerType(), True),
        stype.StructField('month', stype.IntegerType(), True),
        stype.StructField('dn', stype.IntegerType(), True),
        stype.StructField('wday', stype.IntegerType(), True),
        stype.StructField('snap', stype.IntegerType(), True),
        stype.StructField('dept_id', stype.StringType(), True),
        stype.StructField('item_id', stype.StringType(), True),
        stype.StructField('store_id', stype.StringType(), True),
        stype.StructField('sales', stype.DoubleType(), True),
        stype.StructField('flag_ram', stype.IntegerType(), True),
        stype.StructField('Sales_Pred', stype.DoubleType(), True),
    ])

    csv_path = datadir / "Sales5_Ab2011_InklPred.csv"
    print(f"--- Reading: '{csv_path}'")

    sales5: DataFrame = spark.read.csv(str(csv_path), header='true', schema=schema) \
        .withColumn("label", sfunc.col('sales'))

    ppdf = pp(sales5)
    print(f"--- Writing: '{nam}'")

    hlp.writeToDatadirParquet(ppdf, nam)


def preprocessing():
    spark = SparkSession.builder \
        .appName(os.path.basename(__file__)) \
        .getOrCreate()
    prepro(spark, hlp.get_datadir(), "sp5_01")
    spark.stop()


if __name__ == '__main__':
    preprocessing()
