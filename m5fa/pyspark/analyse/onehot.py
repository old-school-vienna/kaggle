from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder \
    .appName("karl01") \
    .config("spark.driver.memory", "12g") \
    .getOrCreate()

df = spark.createDataFrame([
    ("A", "X"),
    ("A", "Y"),
    ("A", "Y"),
    ("B", "Z"),
    ("B", "X"),
    ("A", "Z"),
    ("B", "Y"),
], ["avar", "xvar"])

stages = []

stages += [StringIndexer(inputCol="avar",
                         outputCol="iavar")]
stages += [StringIndexer(inputCol="xvar",
                         outputCol="ixvar")]
stages += [OneHotEncoderEstimator(inputCols=["iavar", "ixvar"],
                                 outputCols=["vavar", "vxvar"])]

pip = Pipeline(stages=stages)
pipm = pip.fit(df)

dft = pipm.transform(df)
dft.show()
