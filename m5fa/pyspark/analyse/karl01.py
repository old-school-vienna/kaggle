import os
from pyspark.shell import spark
from pyspark.sql import DataFrame

datadir: str = os.getenv("DATADIR")

train: DataFrame = spark.read.csv(f"{datadir}/Sales5_Ab2011_InklPred.csv", header=True)

train.show()
