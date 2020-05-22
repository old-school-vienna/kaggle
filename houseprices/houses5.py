import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split

dsall = pd.read_csv("data/Houses5_CategsAsCols_MissingMedianReplaced.csv")
dstr = dsall.loc[dsall['scenario_train'] == 1]
dste = dsall.loc[dsall['scenario_test'] == 0]

non_features = ['Id', 'SalePrice', 'scenario_train', 'scenario_train']

xkeys = [key for key in dstr.keys() if key not in non_features]
print(f"we have {len(xkeys)} features. wow")

X = dstr[xkeys].values

dfy = (dstr[['SalePrice']] * 10).astype(int)
print(dfy.describe())
y = dfy.values
print("y shape", y.shape)
print("y dtype", y.dtype)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)


"""
mds = [None, 1, 2, 3, 10, 100, 200]
data = []
for md in mds:
    scores = []
    ns = range(100)
    for n in ns:
        clf = ensemble.RandomForestClassifier(max_depth=md, n_estimators=1)
        clf.fit(X_train, y_train.ravel())
        s = clf.score(X_test, y_test.ravel())
        scores.append(s)
    data.append(scores)

plt.boxplot(data, labels=["None", "1", "2", "3", "10", "100", "200"])
plt.title("'max_depth' crossvalidation")
plt.show()
"""

ests = [1, 2, 3, 10, 20, 50]
data = []
for est in ests:
    scores = []
    ns = range(100)
    for n in ns:
        clf = ensemble.RandomForestClassifier(max_depth=1, n_estimators=est)
        clf.fit(X_train, y_train.ravel())
        s = clf.score(X_test, y_test.ravel())
        scores.append(s)
    data.append(scores)

plt.boxplot(data, labels=ests)
plt.title("'n_estimators' crossvalidation")
plt.show()
