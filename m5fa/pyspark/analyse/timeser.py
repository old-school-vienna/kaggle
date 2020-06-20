from attr import dataclass
from pyspark import SparkContext
from pyspark.sql import SparkSession


@dataclass
class SomeData:
    anint: int
    astr: str


spark = SparkSession.builder.getOrCreate()
sc: SparkContext = spark.sparkContext
rdd = sc.parallelize([
    SomeData(2, 'hallo'),
    SomeData(5, 'uff'),
])

for d in rdd.collect():
    print(d)
