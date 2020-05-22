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

print("y shape", y.shape)
print("y dtype", y.dtype)

mds = [None, 1, 2, 3, 10, 100, 500]
data = []
for md in mds:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=random.randrange(10000))
    scores = []
    ns = range(100)
    for n in ns:
        clf = ensemble.RandomForestClassifier(max_depth=md, n_estimators=1)
        clf.fit(X_train, y_train.ravel())
        s = clf.score(X_test, y_test.ravel())
        scores.append(s)
    data.append(scores)

plt.boxplot(data, labels=["None", "1", "2", "3", "10", "100", "500"])
plt.title("'max_depth' crossvalidation")
plt.show()

ests = [1, 2, 3, 10, 20, 50]
data = []
for est in ests:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=random.randint(0, 5000))
    scores = []
    ns = range(100)
    for n in ns:
        clf = ensemble.RandomForestClassifier(max_depth=None, n_estimators=est)
        clf.fit(X_train, y_train.ravel())
        s = clf.score(X_test, y_test.ravel())
        scores.append(s)
    data.append(scores)

plt.boxplot(data, labels=ests)
plt.title("'n_estimators' crossvalidation")
plt.show()