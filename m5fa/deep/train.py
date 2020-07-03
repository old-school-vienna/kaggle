from pathlib import Path
from pprint import pprint
from typing import List

import numpy
import pandas as pd
import tensorflow as tf
from pyspark.sql import SparkSession
from tensorflow import keras
from tensorflow.keras import layers

import helpers as hlp
import configuration as cfg

nets = [
    ([], [1.]),
    ([], [0.5, 1.]),
    ([], [1., 1.]),
    ([], [1., 1., 1.]),
    ([], [1., 1., 1., 1.]),
    ([], [1., 1., 1., 1., 1.]),
    ([1.], [1., 1.]),
    ([1.], [2., 1.]),
    ([1.], [3., 1.]),
    ([1., 2.], [1., 1.]),
    ([1., 2., 2.], [1., 1.]),
    ([1., 2., 2.], [0.5, 1.]),
    ([1., 2., 2.], [0.5, 1., 1.]),
]


def nodes(fact: float, n: int) -> int:
    return int(n * fact)


def nnam(net, nin: int, nout: int) -> str:
    def nam1(li, n) -> list:
        def nodes1(f: float) -> str:
            non = nodes(f, n)
            if non > 1000000:
                return f"{int(non / 1000000)}m"
            elif non > 1000:
                return f"{int(non / 1000)}k"
            else:
                return f"{non}"

        return [nodes1(x) for x in li]

    alln = nam1(net[0], nin) + nam1(net[1], nout)
    return '-'.join(alln)


def build_model_gen(num_input: int, num_output: int, net: tuple, stepw: float):
    mo = keras.Sequential()

    mo.add(layers.Dense(num_input, activation='relu', input_shape=[num_input]))
    for f in net[0]:
        n = nodes(f, num_input)
        mo.add(layers.Dense(n, activation='relu'))
    for f in net[1]:
        n = nodes(f, num_output)
        mo.add(layers.Dense(n, activation='relu'))
    mo.add(layers.Dense(num_output))

    optimizer = tf.keras.optimizers.RMSprop(stepw)
    mo.compile(loss='mse',
               optimizer=optimizer,
               metrics=['mae', 'mse'])
    return mo


def trainmulti(sp: SparkSession):
    subs = cfg.subsets[2]
    fnam = cfg.create_fnam(subs[0])

    df: pd.DataFrame = hlp.readFromDatadirParquet(sp, fnam).toPandas()
    df = df.astype(float)

    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    train_data, train_labels = split_data_labels(train_dataset)
    test_data, test_labels = split_data_labels(test_dataset)
    print("----DATA-----------------------------------------------------------------------")
    print(f"-- train data: {train_data.shape}")
    print(f"-- train labels: {train_labels.shape}")
    print(f"-- test data: {test_data.shape}")
    print(f"-- test labels: {test_labels.shape}")
    print("-------------------------------------------------------------------------------")

    stepw = 0.001
    results = []
    for net in nets:
        mse = train_cross(net, stepw, test_data, test_labels, train_data, train_labels)
        results.append((nnam(net, train_data.shape[1], train_labels.shape[1]), mse))

    print("----RESULTS (final)---------------------------------------------------------------")
    for nam, mse in results:
        print(f"-- {nam:30} - {mse:10.4f}")
    print("----------------------------------------------------------------------------------")


def split_data_labels(df: pd.DataFrame) -> tuple:
    allvars = df.keys()

    yvars, xvars = hlp.split_vars(allvars)
    pprint(f"-- predictors X: {xvars}")
    pprint(f"-- labels     y: {yvars}")

    return (df[xvars], df[yvars])


def train_cross(net: tuple, stepw: float,
                test_data: pd.DataFrame, test_labels: pd.DataFrame,
                train_data: pd.DataFrame, train_labels: pd.DataFrame) -> float:
    history, model = train(net, stepw, train_data, train_labels)
    print("----HISTORY------------------------------------------------------------")
    tmses = history.history['mse']
    print(f"-- train mse {len(tmses)} ..., {tmses[-4:]}")
    print("-----------------------------------------------------------------------")
    test_predictions = model.predict(test_data)
    mse = ((test_labels.values - test_predictions) ** 2).mean(axis=0)
    msem = numpy.mean(mse)
    print("----RESULT------------------------------------------------------------")
    print(f"-- mse mean: {msem:10.4f}")
    print("----------------------------------------------------------------------")
    return msem


def train(net: tuple, stepw: float, data: pd.DataFrame, labels: pd.DataFrame) -> tuple:
    nin = data.shape[1]
    nout = labels.shape[1]
    model = build_model_gen(nin, nout, net, stepw=stepw)
    cb = tf.keras.callbacks.EarlyStopping(
        monitor='mse', min_delta=0, patience=0, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )
    history = model.fit(
        data, labels,
        epochs=100, verbose=0,
        callbacks=[cb],
    )
    return history, model


def train_save(sp: SparkSession):
    subs_nam, subs_ids = cfg.subsets[1]
    net = nets[11]
    stepw = 0.001
    fnam = cfg.create_fnam(subs_nam)
    df: pd.DataFrame = hlp.readFromDatadirParquet(sp, fnam).toPandas().astype(float)
    data, labels = split_data_labels(df)
    hist, model = train(net, stepw, data, labels)
    outp = cfg.create_trained_modelPath(subs_nam)
    model.save(str(outp))
    print("----------------------------------------------")
    print(f"-- saved tensorflow model to '{outp}'")


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("train") \
        .getOrCreate()

    # trainmulti(spark)
    train_save(spark)
