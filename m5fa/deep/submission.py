from pyspark.sql import SparkSession, DataFrame as SpDataFrame
import pyspark.sql.functions as F
from tensorflow.python.keras.models import Model as TfModel

import helpers as hlp
import configuration as cfg
from tensorflow import keras
import pandas as pd

subset_id = 2

subs_nam, items = cfg.subsets[subset_id]

sp = SparkSession.builder \
    .appName("submission") \
    .getOrCreate()

fnam = cfg.create_fnam(subs_nam)
df: pd.DataFrame = hlp.readFromDatadirParquet(sp, fnam) \
    .where(F.col("dn").between(1942, 1969)) \
    .drop('Sales_pred', 'sales') \
    .toPandas()

df = df.astype(float)

df_data, df_labels = hlp.split_data_labels(df)

print(df_data)
print(df_data.keys())
print(df_data.dtypes)

model_path = cfg.create_trained_model_path(subs_nam)
print(f"-- loading tf model from {model_path}")
model: TfModel = keras.models.load_model(str(model_path))
pred: pd.DataFrame = model.predict(df_data)

print(pred)
