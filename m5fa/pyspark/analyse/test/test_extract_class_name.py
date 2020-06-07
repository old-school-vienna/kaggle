from pathlib import Path


def extract_classnam(p: Path)-> str:
    return p.name


def test_nulls_none():
    cls = extract_classnam(Path("/tmp/myclass_x1.spark"))
    assert cls == "myclass"

