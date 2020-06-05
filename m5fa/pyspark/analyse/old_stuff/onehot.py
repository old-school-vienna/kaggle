from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder \
    .appName("karl01") \
    .config("spark.driver.memory", "25g") \
    .getOrCreate()

df = spark.createDataFrame([
    ("dummy 0", 0, "A", "X"),
    ("dummy 1", 1, "A", "X"),
    ("dummy 2", 2, "B", "Y"),
    ("dummy 3", 3, "A", "X"),
    ("dummy 4", 4, "D", "Y"),
    ("dummy 5", 5, "B", "X"),
    ("dummy 6", 6, "C", "Z"),
], ["dummy", "i", "avar", "xvar"])

stages = []

catvars = ["avar", "xvar"]
for v in catvars:
    stages += [StringIndexer(inputCol=v,
                             outputCol=f"i{v}")]
ohin = [f"i{v}" for v in catvars]
ohout = [f"v{v}" for v in catvars]
stages += [OneHotEncoderEstimator(inputCols=ohin,
                                  outputCols=ohout)]

stages += [VectorAssembler(inputCols=['vavar', 'vxvar', 'i'],
                           outputCol='features')]

pip = Pipeline(stages=stages)
pipm = pip.fit(df)

dft = pipm.transform(df)
dft.show()
