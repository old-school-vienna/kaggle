import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import random

df_all = pd.read_csv("data/CTrainTestNAsRemoved.csv", sep=";")

cat_keys = [key for key in df_all.keys() if not np.issubdtype(df_all[key].dtype, np.number)]
cont_keys = [key for key in df_all.keys() if key not in cat_keys]
print(f"Cat Columns: {len(cat_keys)}")
print(f"Cat Columns: {len(cont_keys)}")

print(cont_keys)
for key in ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
            'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']:
    df_all[key] = df_all[key].fillna(0.0)

# Show missing
# df_all_cont = df_all[cont_keys]
# nan = (len(df_all_cont.index) - df_all_cont.count())

# print(f"Columns: {len(df_all.keys())}")
# print(f"Columns: {dsall.keys()}")


df_cat = df_all[cat_keys]
df_cont = df_all[cont_keys]

df_cat_dummies = pd.get_dummies(df_cat)
# print(df_cat_dummies)


df_final = df_cat_dummies.join(df_cont)
df_final.to_csv("data/CTrain01.csv")