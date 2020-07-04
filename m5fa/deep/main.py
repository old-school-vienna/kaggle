import os

from pyspark.sql import SparkSession

import preprocessing as pp
import train as tr
import submission as sub

spark = SparkSession.builder \
    .appName(os.path.basename("preporcessing")) \
    .config("spark.sql.pivotMaxValues", 100000) \
    .getOrCreate()
subset_id = 2

# pp.preprocessing(spark, subset_id)
# tr.train_save(spark, subset_id)
# tr.trainmulti(spark, subset_id)
sub.submission(spark, subset_id)
