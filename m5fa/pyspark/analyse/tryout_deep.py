import helpers as hlp
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def number_of_inputs():
    spark = SparkSession.builder \
        .appName("tryout_deep") \
        .getOrCreate()

    row1 = hlp.readFromDatadirParquet(spark, "s5_01") \
        .where(F.col("label").isNotNull()) \
        .rdd \
        .take(1)[0].features
    print(f"number of inputs: {len(row1)}")

number_of_inputs()