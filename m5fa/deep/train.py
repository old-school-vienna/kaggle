from pprint import pprint
from typing import List

import numpy
import pandas as pd
import tensorflow as tf
from pyspark.sql import SparkSession
from tensorflow import keras
from tensorflow.keras import layers

import helpers as hlp


def build_model0(num_input: int, num_output: int):
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


def build_model1(num_input: int, num_output: int):
    mo = keras.Sequential()
    mo.add(layers.Dense(num_input, activation='relu', input_shape=[num_input]))
    mo.add(layers.Dense(num_output, activation='relu'))
    mo.add(layers.Dense(num_output, activation='relu'))
    mo.add(layers.Dense(num_output, activation='relu'))
    mo.add(layers.Dense(num_output, activation='relu'))
    mo.add(layers.Dense(num_output))
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    mo.compile(loss='mse',
               optimizer=optimizer,
               metrics=['mae', 'mse'])
    return mo


def build_model2(num_input: int, num_output: int):
    return build_model_gen(num_input, num_output, ([], [1.0, 1.0, 1.0, 1.0]), 0.001)
    # return build_model_gen(num_input, num_output, [], [1.0], 0.001)


def build_model_gen(num_input: int, num_output: int, net: tuple, stepw: float):
    mo = keras.Sequential()

    mo.add(layers.Dense(num_input, activation='relu', input_shape=[num_input]))
    for f in net[0]:
        n = int(f * num_input)
        mo.add(layers.Dense(n, activation='relu'))
    for f in net[1]:
        n = int(f * num_input)
        mo.add(layers.Dense(n, activation='relu'))
    mo.add(layers.Dense(num_output))

    optimizer = tf.keras.optimizers.RMSprop(stepw)
    mo.compile(loss='mse',
               optimizer=optimizer,
               metrics=['mae', 'mse'])
    return mo


def train(sp: SparkSession):
    df: pd.DataFrame = hlp.readFromDatadirParquet(sp, 'sp5_02').toPandas()
    df = df.astype(float)

    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    allvars = df.keys()
    xvars = ['dn', 'flag_ram', 'month', 'snap', 'vdept_id_0', 'vdept_id_1',
             'vwday_0', 'vwday_1', 'vwday_2', 'vwday_3', 'vwday_4', 'vwday_5', 'year']
    yvars = [x for x in allvars if x not in xvars]
    pprint(f"-- predictors X: {xvars}")
    pprint(f"-- labels     y: {yvars}")

    train_data = train_dataset[xvars]
    train_labels = train_dataset[yvars]
    test_data = test_dataset[xvars]
    test_labels = test_dataset[yvars]
    print("----DATA-----------------------------------------------------------------------")
    print(f"-- train data: {train_data.shape}")
    print(f"-- train labels: {train_labels.shape}")
    print(f"-- test data: {test_data.shape}")
    print(f"-- test labels: {test_labels.shape}")
    print("-------------------------------------------------------------------------------")

    epochs = 1000
    cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )

    model = build_model2(len(xvars), len(yvars))
    history = model.fit(
        train_data, train_labels,
        epochs=epochs, validation_split=0.2, verbose=0,
        callbacks=[cb],
    )
    print("----HISTORY------------------------------------------------------------")
    tmses = history.history['mse']
    print(f"-- train mse {len(tmses)} {tmses}")
    print("-----------------------------------------------------------------------")

    test_predictions = model.predict(test_data)

    mse = ((test_labels.values - test_predictions) ** 2).mean(axis=0)
    msem = numpy.mean(mse)
    mses = numpy.sum(mse)
    print("----RESULT------------------------------------------------------------")
    print(f"-- mse mean: {msem:10.4f}")
    print(f"-- mse sum:  {mses:10.4f}")
    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("train") \
        .getOrCreate()

    # df = hlp.readFromDatadirParquet(spark, 'sp5_02')
    # df.show()

    train(spark)
