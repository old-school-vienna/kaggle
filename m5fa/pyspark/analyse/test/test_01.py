from pyspark import Row

from karl03 import nulls


def test_nulls_none():
    d = {
        'a': 1,
        'b': 'hallo'
    }
    r = Row(**d)
    assert nulls(r)['nullcnt'] == 0, "there are no 'None'"


def test_nulls_a():
    d = {
        'a': None,
        'b': 'hallo'
    }
    r = Row(**d)
    assert nulls(r)['nullcnt'] == 1, "there is 1 'None'"


def test_nulls_b():
    d = {
        'a': 2,
        'b': None
    }
    r = Row(**d)
    assert nulls(r)['nullcnt'] == 1, "there is 1 'None'"


def test_nulls_all():
    d = {
        'a': None,
        'b': None
    }
    r = Row(**d)
    assert nulls(r)['nullcnt'] == 2, "there are 2 'None'"
