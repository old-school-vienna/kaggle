from pprint import pprint

from pyspark import Row
from pyspark.ml.linalg import Vectors

from helpers import one_hot_row


def test_one_hot():
    d1 = {
        'a': 100,
        'b': 'hallo',
        'c': Vectors.dense([0.5, 6, 0.7]),
        'd': Vectors.sparse(4, {1: 2.0, 3: 5})
    }
    r1 = Row(**d1)
    r2 = one_hot_row(r1)
    assert (r2['a'] == 100)
    assert (r2['b'] == 'hallo')
    assert (r2['c_0'] == 0.5)
    assert (r2['c_1'] == 6)
    assert (r2['c_2'] == 0.7)
    assert (r2['d_0'] == 0)
    assert (r2['d_1'] == 2.0)
    assert (r2['d_2'] == 0)
    assert (r2['d_3'] == 5)
