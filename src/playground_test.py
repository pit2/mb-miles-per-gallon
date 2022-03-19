from playground import map_
from playground import false_map
from playground import my_div
from playground import my_false_div
import pytest


def test_map_():

    results = []
    expects = []
    results.append(map_([1, 1, 1], lambda x: x))
    expects.append([1, 1, 1])
    results.append(map_(range(1, 10), lambda x: x))
    expects.append(range(1, 10))
    results.append(map_(range(1, 10), lambda x: x + 1))
    expects.append(range(2, 11))
    results.append(map_([-4, 12, 5.5, 0], lambda x: x * 3))
    expects.append([-12, 36, 16.5, 0])
    results.append(map_([], lambda x: x))
    expects.append([])
    for res, exp in zip(results, expects):
        assert all([a == b for a, b in zip(res, exp)])


def test_false_map():

    res = false_map([1, 1, 1], lambda x: x)
    exp = [1, 1, 1]
    assert all([a == b for a, b in zip(res, exp)])
    res = false_map(range(1, 10), lambda x: x + 1)
    exp = range(2, 11)
    assert all([a == b for a, b in zip(res, exp)])
    res = false_map([-4, 12, 5.5, 0], lambda x: x * 3)
    exp = [-12, 36, 16.5, 0]
    assert all([a == b for a, b in zip(res, exp)])
    res = false_map([], lambda x: x)
    exp = ([])
    assert all([a == b for a, b in zip(res, exp)])


def test_my_div():
    assert my_div(12, 3) == 4
    assert my_div(12, 5) == 2
    assert my_div(3, 12) == 0
    with pytest.raises(ZeroDivisionError):
        my_div(12, 0)


def test_my_false_div():
    assert my_false_div(12, 3) == 4
    assert my_false_div(12, 5) == 2
    assert my_false_div(3, 12) == 0
    with pytest.raises(ZeroDivisionError):
        my_false_div(12, 0)
