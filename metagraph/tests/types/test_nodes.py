import pytest

grblas = pytest.importorskip("grblas")

from metagraph.plugins.python.types import PythonNodeMapType
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyNodeSet
from metagraph.plugins.graphblas.types import GrblasNodeMap, GrblasNodeSet
from metagraph import NodeLabels
import numpy as np
from grblas import Vector


def test_python():
    PythonNodeMapType.assert_equal(
        {"A": 1, "B": 2, "C": 3},
        {"A": 1, "B": 2, "C": 3},
        {"dtype": "int"},
        {"dtype": "int"},
        {},
        {},
    )
    PythonNodeMapType.assert_equal(
        {"A": 1, "C": 3.333333333333333333333333, "B": 2},
        {"A": 1, "C": 3.333333333333333333333334, "B": 2 + 1e-9},
        {"dtype": "float"},
        {"dtype": "float"},
        {},
        {},
    )
    with pytest.raises(AssertionError):
        PythonNodeMapType.assert_equal(
            {"A": 1}, {"A": 1, "B": 2}, {"dtype": "int"}, {"dtype": "int"}, {}, {},
        )
    with pytest.raises(AssertionError):
        PythonNodeMapType.assert_equal(
            {"A": 1, "B": 22},
            {"A": 1, "B": 2},
            {"dtype": "int"},
            {"dtype": "int"},
            {},
            {},
        )
    with pytest.raises(AssertionError):
        PythonNodeMapType.assert_equal(
            {"A": 1.1}, {"A": 1}, {"dtype": "float"}, {"dtype": "int"}, {}, {},
        )
    with pytest.raises(TypeError, match="Unable to compute dtype"):
        PythonNodeMapType.compute_abstract_properties({0: 3 + 4j, 1: 5 - 2j}, {"dtype"})


def test_numpy():
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(np.array([1, 3, 5, 7, 9])),
        NumpyNodeMap(np.array([1, 3, 5, 7, 9])),
        {},
        {},
        {},
        {},
    )
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(np.array([1, 3, 5.5555555555555555555, 7, 9])),
        NumpyNodeMap(np.array([1, 3, 5.5555555555555555556, 7, 9 + 1e-9])),
        {},
        {},
        {},
        {},
    )
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(np.array([1, 3, 5, 7, 9]), [14, 2, 7, 8, 20]),
        NumpyNodeMap(np.array([1, 3, 5, 7, 9]), [14, 2, 7, 8, 20]),
        {},
        {},
        {},
        {},
    )
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(np.array([1, 3, 5, 7, 9]), [0, 2, 4, 6, 8]),
        NumpyNodeMap.from_mask(
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            np.array([True, False, True, False, True, False, True, False, True, False]),
        ),
        {},
        {},
        {},
        {},
    )
    with pytest.raises(AssertionError):
        NumpyNodeMap.Type.assert_equal(
            NumpyNodeMap(np.array([1, 3, 5, 7, 9])),
            NumpyNodeMap(np.array([1, 3, 5, 7, 9, 11])),
            {},
            {},
            {},
            {},
        )
    with pytest.raises(AssertionError):
        NumpyNodeMap.Type.assert_equal(
            NumpyNodeMap(np.array([1, 3, 5, 7, 9]), np.array([14, 2, 7, 8, 20])),
            NumpyNodeMap(np.array([1, 3, 5, 7, 9]), np.array([2, 7, 8, 14, 20])),
            {},
            {},
            {},
            {},
        )

    # Exercise NumpyNodeSet
    with pytest.raises(TypeError, match="Invalid number of dimensions: 2"):
        NumpyNodeSet(np.array([[1, 2, 3], [4, 5, 6]]))
    with pytest.raises(TypeError, match="Invalid dtype for NodeSet"):
        NumpyNodeSet(np.array([1.1, 2.2, 3.3]))
    # Handle duplicates
    x = NumpyNodeSet([1, 1, 3, 4, 1, 2, 1])
    assert len(x) == 4
    assert 1 in x
    assert [2, 3, 4] in x

    # Exercise NumpyNodeMap
    with pytest.raises(TypeError, match="Invalid number of dimensions: 2"):
        NumpyNodeMap(np.array([[1, 2, 3], [4, 5, 6]]))
    with pytest.raises(TypeError, match="Nodes must be same shape and size as data"):
        NumpyNodeMap([1, 2, 3, 4], nodes=[1, 2, 3])
    with pytest.raises(TypeError, match="Invalid dtype for nodes"):
        NumpyNodeMap([1, 2, 3, 4], nodes=[1.1, 2.2, 3.3, 4.4])
    with pytest.raises(TypeError, match="Duplicate node ids found"):
        NumpyNodeMap([1, 2, 3, 4], nodes=[1, 1, 1, 2])
    y = NumpyNodeMap([1.1, 2.2, 3.3, 4.4], nodes=[5, 6, 7, 22])
    assert len(y) == 4
    assert 22 in y
    assert [5, 6, 7] in y
    assert y[5] == 1.1
    with pytest.raises(KeyError, match="is not in the NodeMap"):
        y[17]
    with pytest.raises(KeyError, match="are not all in the NodeMap"):
        y[[7, 8, 9]]


def test_graphblas():
    GrblasNodeMap.Type.assert_equal(
        GrblasNodeMap(Vector.from_values([0, 1, 3, 4], [1, 2, 3, 4])),
        GrblasNodeMap(Vector.from_values([0, 1, 3, 4], [1, 2, 3, 4])),
        {},
        {},
        {},
        {},
    )
    GrblasNodeMap.Type.assert_equal(
        GrblasNodeMap(
            Vector.from_values([0, 1, 3, 4], [1.0, 2.0, 3.333333333333333333, 4.0])
        ),
        GrblasNodeMap(
            Vector.from_values([0, 1, 3, 4], [1.0, 2.0, 3.333333333333333334, 4 + 1e-9])
        ),
        {},
        {},
        {},
        {},
    )
    with pytest.raises(AssertionError):
        GrblasNodeMap.Type.assert_equal(
            GrblasNodeMap(Vector.from_values([0, 1, 3, 4], [1, 2, 3, 4])),
            GrblasNodeMap(Vector.from_values([0, 1, 2, 4], [1, 2, 3, 4])),
            {},
            {},
            {},
            {},
        )
    with pytest.raises(AssertionError):
        GrblasNodeMap.Type.assert_equal(
            GrblasNodeMap(Vector.from_values([0, 1, 2], [1, 2, 3])),
            GrblasNodeMap(Vector.from_values([0, 1, 2, 3], [1, 2, 3, 4])),
            {},
            {},
            {},
            {},
        )

    # Exercise GrblasNodeSet
    x = GrblasNodeSet(Vector.from_values([0, 1, 3], [1, 1, 1]))
    assert len(x) == 3
    assert 3 in x
    assert 2 not in x

    # Exercise GrblasNodeMap
    y = GrblasNodeMap(Vector.from_values([0, 1, 3], [1.1, 2.2, 3.3]))
    assert len(y) == 3
    assert 3 in y
    assert 2 not in y
    assert y[3] == 3.3
