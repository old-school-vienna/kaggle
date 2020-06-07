from dataclasses import dataclass
from pprint import pprint

from pyspark.ml import Transformer
from pyspark.sql import SparkSession

import common as cm
import helpers as hlp


@dataclass
class Model:
    desc: str
    id: str


models = [
    Model("glr poisson identity", "glrpi"),
    Model("glr gaussian identity", "glrgi"),
    Model("glr poisson log", "glrpi"),
]


def load_model(model: Model) -> Transformer:
    fnam = cm.fnam(model.id)
    return hlp.load_model(hlp.get_datadir(), fnam)


spark = SparkSession.builder \
    .appName("submission") \
    .getOrCreate()

for m in models:
    t: Transformer = load_model(m)
    pmap = t.extractParamMap()
    print(f"Loaded {m}")
    pprint(pmap)