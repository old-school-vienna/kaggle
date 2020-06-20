import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from tensorflow import keras
from tensorflow.keras import layers

import helpers as hlp


def number_of_inputs():
    spark = SparkSession.builder \
        .appName("tryout_deep") \
        .getOrCreate()

    row1 = hlp.readFromDatadirParquet(spark, "s5_01") \
        .where(F.col("label").isNotNull()) \
        .rdd \
        .take(1)[0].features
    print(f"number of inputs: {len(row1)}")


def make_nn():
    model = keras.Sequential()
    model.add(keras.Input(shape=(3074,)))
    model.add(layers.Dense(1500, activation="relu"))
    model.add(layers.Dense(1, activation="relu"))

    model.summary()


make_nn()
