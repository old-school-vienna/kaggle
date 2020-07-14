from datetime import datetime
import os
from pathlib import Path
import pandas as pd
import helpers as hlp

"""
Filenames in DATADIR

test.csv
item_categories.csv
shops.csv
items.csv
sample_submission.csv
sales_train.csv

sales_train_dn.csv

   22.165 items
2.935.849 training data
       59 shops
   0 - 34 months
 0 - 1033 days
"""

dt_start: datetime.date = datetime.strptime("1.1.2013", '%d.%m.%Y').date()


def analyse_sales_train(data_dir: Path):
    df = pd.read_csv(data_dir / "sales_train_dn.csv")
    print("-- df shape", df.shape)
    print("-- df keys", df.keys())
    print(df.head(200))
    print("dn min", df.dn.min())
    print("dn max", df.dn.max())


def analyse_items(data_dir: Path):
    df = pd.read_csv(data_dir / "items.csv")[['item_id', 'item_category_id']]
    print("-- df shape", df.shape)

    # print("-- describe -------------------------------------")
    # print(df.describe())

    print("-- df keys", df.keys())
    print("-- df", df)


dd = hlp.dd()
analyse_sales_train(dd)
