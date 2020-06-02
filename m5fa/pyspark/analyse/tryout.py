from pyspark import SparkContext
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName(__name__) \
    .getOrCreate()

sc: SparkContext = spark.sparkContext

rdd_long = sc.parallelize(range(0, 10000000))

l = rdd_long.take(20)
print(f"type from take: {type(l)}")

rdd_short = sc.parallelize(l)\
    .filter(lambda x: x % 2 == 0)

print(f"len of short: {rdd_short.collect()}")