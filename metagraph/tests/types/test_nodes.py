import pytest

grblas = pytest.importorskip("grblas")

from metagraph.plugins.python.types import PythonNodeMap
from metagraph.plugins.numpy.types import NumpyNodeMap
from metagraph.plugins.graphblas.types import GrblasNodeMap
from metagraph import NodeLabels
import numpy as np
from grblas import Vector


def test_python():
    PythonNodeMap.Type.assert_equal(
        PythonNodeMap({"A": 1, "B": 2, "C": 3}),
        PythonNodeMap({"A": 1, "B": 2, "C": 3}),
        {"dtype": "int"},
        {"dtype": "int"},
        {},
        {},
    )
    PythonNodeMap.Type.assert_equal(
        PythonNodeMap({"A": 1, "C": 3.333333333333333333333333, "B": 2}),
        PythonNodeMap({"A": 1, "C": 3.333333333333333333333334, "B": 2 + 1e-9}),
        {"dtype": "float"},
        {"dtype": "float"},
        {},
        {},
    )
    with pytest.raises(AssertionError):
        PythonNodeMap.Type.assert_equal(
            PythonNodeMap({"A": 1}),
            PythonNodeMap({"A": 1, "B": 2}),
            {"dtype": "int"},
            {"dtype": "int"},
            {},
            {},
        )
    with pytest.raises(AssertionError):
        PythonNodeMap.Type.assert_equal(
            PythonNodeMap({"A": 1, "B": 22}),
            PythonNodeMap({"A": 1, "B": 2}),
            {"dtype": "int"},
            {"dtype": "int"},
            {},
            {},
        )
    with pytest.raises(AssertionError):
        PythonNodeMap.Type.assert_equal(
            PythonNodeMap({"A": 1.1}),
            PythonNodeMap({"A": 1}),
            {"dtype": "float"},
            {"dtype": "int"},
            {},
            {},
        )


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
    with pytest.raises(AssertionError):
        NumpyNodeMap.Type.assert_equal(
            NumpyNodeMap(np.array([1, 3, 5, 7, 9])),
            NumpyNodeMap(np.array([1, 3, 5, 7, 9, 11])),
            {},
            {},
            {},
            {},
        )
    # Missing value should not affect equality
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(
            np.array([1, -1, -1, 4, 5, -1]),
            mask=np.array([True, False, False, True, True, False]),
        ),
        NumpyNodeMap(
            np.array([1, 0, 0, 4, 5, 0]),
            mask=np.array([True, False, False, True, True, False]),
        ),
        {},
        {},
        {},
        {},
    )


def test_numpy_compact():
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(np.array([1, 3, 5]), node_ids={0: 0, 240: 1, 968: 2}),
        NumpyNodeMap(np.array([1, 3, 5]), node_ids=np.array([0, 240, 968])),
        {},
        {},
        {},
        {},
    )
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(
            np.array([1, 3, 5.5555555555555555555]), node_ids={0: 0, 1: 1, 2: 2}
        ),
        NumpyNodeMap(
            np.array([1, 3 + 1e-9, 5.5555555555555555556]), node_ids={0: 0, 1: 1, 2: 2}
        ),
        {},
        {},
        {},
        {},
    )
    with pytest.raises(AssertionError):
        NumpyNodeMap.Type.assert_equal(
            NumpyNodeMap(np.array([1, 3, 5]), node_ids={0: 0, 1: 1, 2: 2}),
            NumpyNodeMap(np.array([1, 3, 5, 7]), node_ids={0: 0, 1: 1, 2: 2, 3: 3}),
            {},
            {},
            {},
            {},
        )
    # Non-monotonic
    with pytest.raises(TypeError):
        NumpyNodeMap(np.array([5, 1, 3]), node_ids=np.array([7, 0, 2])),


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
