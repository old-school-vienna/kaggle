from pyspark import SparkContext
from pyspark.ml.regression import GeneralizedLinearRegression

import helpers as hlp


def test_simple():
    cls = hlp.classnam_from_filenam("mod.x__myclass__x1.spark")
    assert cls == hlp.Classnam("mod.x", "myclass")


def test_with_upper():
    cls = hlp.classnam_from_filenam("mod.x__MyClass__x1.spark")
    assert cls == hlp.Classnam("mod.x", "MyClass")


def test_with_unders():
    cls = hlp.classnam_from_filenam("mod.x__my_class__x1.spark")
    assert cls == hlp.Classnam("mod.x", "my_class")


def test_create_nam_glr():
    SparkContext("local", "first app")
    a = GeneralizedLinearRegression()
    cn = hlp.classnam_from_obj(a)
    cn = hlp.create_nam_from_classname(cn, "x.spark")
    assert cn == "pyspark.ml.regression__GeneralizedLinearRegression__x.spark"


def test_file_for_other_class_exists_not():
    fnams = ["x__y__a.sp"]
    cn = hlp.Classnam("x", "y")
    assert hlp.file_for_other_class_exists(fnams, cn, 'a.sp') is False


def test_file_for_other_class_exists_forclass():
    fnams = ["x__y__a.sp"]
    cn = hlp.Classnam("x", "y1")
    assert hlp.file_for_other_class_exists(fnams, cn, 'a.sp') is True


def test_file_for_other_class_exists_formod():
    fnams = ["x__y__a.sp"]
    cn = hlp.Classnam("x1", "y")
    assert hlp.file_for_other_class_exists(fnams, cn, 'a.sp') is True


def test_file_for_other_class_exists_forall():
    fnams = ["x__y__a.sp"]
    cn = hlp.Classnam("x1", "y1")
    assert hlp.file_for_other_class_exists(fnams, cn, 'a.sp') is True
