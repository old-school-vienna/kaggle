import pandas as pd
import tensorflow as tf
from pprint import pprint
from pyspark.sql import SparkSession
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs

import helpers as hlp


def build_model():
    mo = keras.Sequential([
        layers.Dense(15, activation='relu', input_shape=[12]),
        layers.Dense(15, activation='relu'),
        layers.Dense(30)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    mo.compile(loss='mse',
               optimizer=optimizer,
               metrics=['mae', 'mse'])
    return mo


def train(spark: SparkSession):
    xvars = ['FOODS_1_001_CA_1', 'FOODS_1_001_CA_2', 'FOODS_1_001_CA_3',
             'FOODS_1_001_CA_4', 'FOODS_1_001_TX_1', 'FOODS_1_001_TX_2',
             'FOODS_1_001_TX_3', 'FOODS_1_001_WI_1', 'FOODS_1_001_WI_2',
             'FOODS_1_001_WI_3', 'HOBBIES_1_021_CA_1', 'HOBBIES_1_021_CA_2',
             'HOBBIES_1_021_CA_3', 'HOBBIES_1_021_CA_4', 'HOBBIES_1_021_TX_1',
             'HOBBIES_1_021_TX_2', 'HOBBIES_1_021_TX_3', 'HOBBIES_1_021_WI_1',
             'HOBBIES_1_021_WI_2', 'HOBBIES_1_021_WI_3', 'HOUSEHOLD_2_491_CA_1',
             'HOUSEHOLD_2_491_CA_2', 'HOUSEHOLD_2_491_CA_3', 'HOUSEHOLD_2_491_CA_4',
             'HOUSEHOLD_2_491_TX_1', 'HOUSEHOLD_2_491_TX_2', 'HOUSEHOLD_2_491_TX_3',
             'HOUSEHOLD_2_491_WI_1', 'HOUSEHOLD_2_491_WI_2', 'HOUSEHOLD_2_491_WI_3']
    yvars = ['dn', 'flag_ram', 'month', 'snap', 'vdept_id_0', 'vdept_id_1',
             'vwday_0', 'vwday_1', 'vwday_2', 'vwday_3', 'vwday_4', 'vwday_5', 'year']

    print(f"x (predictors) {len(xvars)}")
    print(f"y (labels) {len(yvars)}")

    df: pd.DataFrame = hlp.readFromDatadirParquet(spark, 'sp5_02').toPandas()

    df = df.astype(float)

    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)
    print(train_dataset.shape)
    print(test_dataset.shape)

    model = build_model()

    model.summary()

    train_data = train_dataset[xvars]
    train_labels = train_dataset[yvars]
    test_data = test_dataset[xvars]
    test_labels = test_dataset[yvars]
    print("data")
    print(train_data.shape)
    print(test_data.shape)
    print("labels")
    print(train_labels.shape)
    print(test_labels.shape)

    EPOCHS = 1000

    history = model.fit(
        train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])
    pprint(history)


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("train") \
        .getOrCreate()

    # df = hlp.readFromDatadirParquet(spark, 'sp5_02')
    # df.show()

    train(spark)
