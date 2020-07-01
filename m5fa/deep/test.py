from pprint import pprint

import pytest
from pyspark import Row
from pyspark.ml.linalg import Vectors
import helpers as hlp

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


def test_is_label_LABEL():
    assert hlp._is_label_var('HOUSEHOLD_1_022_CA_2') == True
    assert hlp._is_label_var('HOBBIES_2_145_WI_3') == True
    assert hlp._is_label_var('HOBBIES_2_146_CA_1') == True
    assert hlp._is_label_var('HOBBIES_2_146_CA_2') == True
    assert hlp._is_label_var('HOBBIES_2_146_CA_3') == True
    assert hlp._is_label_var('HOBBIES_2_146_CA_4') == True


def test_is_label_NOLABEL():
    assert hlp._is_label_var("laksjd") == False
    assert hlp._is_label_var('dn') == False
    assert hlp._is_label_var('flag_ram') == False
    assert hlp._is_label_var('month') == False
    assert hlp._is_label_var('snap') == False
    assert hlp._is_label_var('vdept_id_0') == False
    assert hlp._is_label_var('vwday_0') == False
    assert hlp._is_label_var('vwday_1') == False
    assert hlp._is_label_var('vwday_2') == False
    assert hlp._is_label_var('vwday_3') == False
    assert hlp._is_label_var('vwday_4') == False
    assert hlp._is_label_var('vwday_5') == False
    assert hlp._is_label_var('year') == False


def test_split():
    vars = ['flag_ram', 'vdept_id_0', 'HOBBIES_2_146_CA_4', 'HOUSEHOLD_1_022_CA_2']
    x, y = hlp.split_vars(vars)
    assert x == ['flag_ram', 'vdept_id_0']
    assert y == ['HOBBIES_2_146_CA_4', 'HOUSEHOLD_1_022_CA_2']
