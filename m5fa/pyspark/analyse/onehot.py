from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder \
    .appName("karl01") \
    .config("spark.driver.memory", "12g") \
    .getOrCreate()

df = spark.createDataFrame([
    ("dummy 0", 0, "A", "X"),
    ("dummy 1", 1, "A", "X"),
    ("dummy 2", 2, "B", "Y"),
    ("dummy 3", 3, "A", "X"),
    ("dummy 4", 4, "A", "Y"),
    ("dummy 5", 5, "B", "X"),
    ("dummy 6", 6, "A", "Z"),
], ["dummy", "i", "avar", "xvar"])

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
