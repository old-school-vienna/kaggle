import pandas as pd
import helpers as hlp

dd = hlp.dd()
df = pd.read_csv(dd / "sales_train.csv")
print(df)

df['dn'] = df.apply(lambda row: hlp.to_ds(row.date), axis=1)
print(df)

out_nam = "sales_train_dn.csv"
out_path = dd / out_nam
print("-- writing to", out_path)
df.to_csv(out_path)
print("-- wrote to", out_path)
