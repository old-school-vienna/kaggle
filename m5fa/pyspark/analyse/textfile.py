from pyspark.shell import spark, sc

from pyspark.sql.functions import DataFrame, explode, split

sc.setLogLevel("ERROR")
textFile: DataFrame = spark.read.text("data/test.txt")

print(f"Explain textFile: {textFile.explain()}")
print(f"Count   textFile: {textFile.count()}")
print(f"Columns textFile: {textFile.columns}")
print(f"Schema  textFile: {textFile.schema}")
lines = textFile.value
slines = split(lines, "\s+")
a = explode(slines).alias("word")
wordCounts: DataFrame = textFile.select(a)\
    .groupBy("word") \
    .count() \
    .sort("word", ascending=False) 
print(f"words   textFile: {wordCounts.collect()}")
