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

df_train = df_final[df_all.SalePrice.notna()]
df_test = df_final[df_all.SalePrice.isna()]

print(df_train.shape)
print(df_test.shape)

non_features = ['Id', 'SalePrice']

xkeys = [key for key in df_final.keys() if key not in non_features]

X = df_train[xkeys].values
y = df_train['SalePrice'].values

print("X shape", X.shape)
print("y dtype", y.dtype)

X1 = df_test[xkeys].values
print("X1 shape", X1.shape)

clf = ensemble.RandomForestClassifier(max_depth=None, n_estimators=1)
clf.fit(X, y.ravel())

p = clf.predict(X1)
t = df_train['SalePrice']
plt.boxplot([p, t], labels=["pred", "train"])
plt.title("'predicted / training sale price")
plt.show()

df_subm: pd.DataFrame = df_test[['Id']].join(pd.DataFrame(p, columns=["SalePrice"]))
print(df_subm.shape)
print(df_subm.keys())
df_sub.to
