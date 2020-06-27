from pprint import pprint

import numpy
import pandas as pd
import tensorflow as tf
from pyspark.sql import SparkSession
from tensorflow import keras
from tensorflow.keras import layers

import helpers as hlp


def build_model(num_input: int, num_output: int):
    mo = keras.Sequential([
        layers.Dense(num_input, activation='relu', input_shape=[num_input]),
        layers.Dense(num_output, activation='relu'),
        layers.Dense(num_output, activation='relu'),
        layers.Dense(num_output, activation='relu'),
        layers.Dense(num_output, activation='relu'),
        layers.Dense(num_output)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    mo.compile(loss='mse',
               optimizer=optimizer,
               metrics=['mae', 'mse'])
    return mo


def train(spark: SparkSession):
    df: pd.DataFrame = hlp.readFromDatadirParquet(spark, 'sp5_02').toPandas()

    df = df.astype(float)

    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    print(train_dataset.shape)
    print(test_dataset.shape)

    allvars = df.keys()
    xvars = ['dn', 'flag_ram', 'month', 'snap', 'vdept_id_0', 'vdept_id_1',
             'vwday_0', 'vwday_1', 'vwday_2', 'vwday_3', 'vwday_4', 'vwday_5', 'year']
    yvars = [x for x in allvars if x not in xvars]
    pprint(xvars)
    pprint(yvars)


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

    epochs = 1000
    cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )

    model = build_model(len(xvars), len(yvars))
    history = model.fit(
        train_data, train_labels,
        epochs=epochs, validation_split=0.2, verbose=0,
        callbacks=[cb],
    )
    print("----HISTORY------------------------------------------------------------")
    for h in history.history['mse']:
        pprint(h)
    print("-----------------------------------------------------------------------")
    test_predictions = model.predict(test_data)

    pprint(test_labels.shape)
    pprint(test_predictions.shape)

    mse = ((test_labels.values - test_predictions) ** 2).mean(axis=0)
    print(f"-- mse {numpy.mean(mse)}")


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("train") \
        .getOrCreate()

    # df = hlp.readFromDatadirParquet(spark, 'sp5_02')
    # df.show()

    train(spark)
