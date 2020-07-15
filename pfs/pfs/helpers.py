import os
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


def pivot(df: pd.DataFrame, grp_vars: List[str], col: str, val: str) -> pd.DataFrame:
    grpd = df.groupby(grp_vars).first()
    return grpd.pivot_table(index=grp_vars, columns=col, values=val, fill_value=0.0)


def dd() -> Path:
    datadir = os.getenv("DATADIR")
    if datadir is None:
        raise RuntimeError("Environment variable DATADIR not defined")
    datadir_path = Path(datadir)
    if not datadir_path.exists():
        raise RuntimeError(f"Directory {datadir_path} does not exist")
    return datadir_path


_dt_start: datetime.date = datetime.strptime("1.1.2013", '%d.%m.%Y').date()


def to_ds(date: str) -> int:
    dt: datetime.date = datetime.strptime(date, '%d.%m.%Y').date()
    diff = dt - _dt_start
    return diff.days
