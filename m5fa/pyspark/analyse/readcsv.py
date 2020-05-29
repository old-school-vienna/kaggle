import os
from typing import List

from pyspark import RDD
from pyspark.shell import spark, sc
from pyspark.sql import DataFrame


def tvars(prefix: str, fr: int, to: int) -> RDD:
    data = range(fr, to + 1)
    distData: RDD = sc.parallelize(data)
    return distData \
        .map(lambda x: (x, f"{prefix}{x}")) \
        .sortBy(lambda x: x)


def varnams(vars: RDD) -> List[str]:
    return vars.map(lambda x: x[1]).collect()


def main():
    datadir: str = os.getenv("DATADIR")
    t_vars_all = tvars("d_", 1, 1913)
    t_vars_small = tvars("d_", 1, 15)

    grp_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    print(grp_vars)

    train: DataFrame = spark.read.csv(f"{datadir}/sales_train_validation.csv", header=True)

    train1 = train[grp_vars] \
        .sort('item_id', 'state_id')

    train2 = train[varnams(t_vars_small)] \
        .sort('item_id', 'state_id')

    train2.show()


if __name__ == "__main__":
    # execute only if run as a script
    main()
