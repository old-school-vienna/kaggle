from pprint import pprint
from typing import List

import numpy
import pandas as pd
import tensorflow as tf
from pyspark.sql import SparkSession
from tensorflow import keras
from tensorflow.keras import layers

import helpers as hlp


def build_model(num_input: int, num_output: int, ifac_in: List[float], ifac_out: List[float], stepw: float):
    mo = keras.Sequential()

    mo.add(layers.Dense(num_input, activation='relu', input_shape=[num_input]))
    for f in ifac_in:
        n = int(f * num_input)
        mo.add(layers.Dense(n, activation='relu'))
    for f in ifac_out:
        n = int(f * num_input)
        mo.add(layers.Dense(n, activation='relu'))
    mo.add(layers.Dense(num_output))

    optimizer = tf.keras.optimizers.RMSprop(stepw)
    mo.compile(loss='mse',
               optimizer=optimizer,
               metrics=['mae', 'mse'])
    return mo


def train1(net: tuple, stepw: float, data: dict) -> float:
    model = build_model(len(data['xvars']), len(data['yvars']), net[0], net[1], stepw)

    cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )

    history = model.fit(
        data['train_data'], data['train_labels'],
        epochs=1000, validation_split=0.2, verbose=0,
        callbacks=[cb],
    )
    print("----HISTORY------------------------------------------------------------")
    for h in history.history['mse']:
        pprint(h)
    print("-----------------------------------------------------------------------")
    test_predictions = model.predict(data['test_data'])

    pprint(data['test_labels'].shape)
    pprint(test_predictions.shape)

    error = ((data['test_labels'].values - test_predictions) ** 2).mean(axis=0)[0]
    nam = nnam(net)
    print(f"---- mse of {nam}: {error:.3f}")
    return error


def nnam(net) -> str:
    def nam1(list) -> str:
        fs = [f"{x:.2f}" for x in list]
        return "-".join(fs)

    return f"|{nam1(net[0])}|{nam1(net[1])}|"


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

    data = {
        'xvars': xvars,
        'yvars': yvars,
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
    }

    stepws = [0.005, 0.001, 0.0001, 0.00001]
    nets = [
        ([1.0], [1.0]),
        ([1.0, 1.5], [1.0, 1.0]),
        ([1.0, 1.5], [0.5, 0.5, 1.0, 1.0, 1.0]),
        ([1.0, 1.5, 2.0], [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]),
        ([1.0, 1.5, 2.0], [0.1, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ]
    results = []
    for stepw in stepws:
        for net in nets:
            err = [train1(net, stepw, data) for net in nets]
            nam = nnam(net)
            results.append((stepw, nam, err))

    print("-------RESULTS-------------------------------------------------")
    for stepw, nam, err in results:
        print(f"{stepw:10.5f} {nam:50} - {err:10.4f}")


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("train") \
        .getOrCreate()

    # df = hlp.readFromDatadirParquet(spark, 'sp5_02')
    # df.show()

    train(spark)
