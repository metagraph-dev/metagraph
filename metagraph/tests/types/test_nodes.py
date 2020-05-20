import pytest
from metagraph.plugins.python.types import PythonNodeMap
from metagraph.plugins.numpy.types import NumpyNodeMap, CompactNumpyNodeMap
from metagraph.plugins.graphblas.types import GrblasNodeMap
from metagraph import IndexedNodes
import numpy as np
from grblas import Vector


def test_python():
    PythonNodeMap.Type.assert_equal(
        PythonNodeMap({"A": 1, "B": 2, "C": 3}), PythonNodeMap({"A": 1, "B": 2, "C": 3})
    )
    PythonNodeMap.Type.assert_equal(
        PythonNodeMap({"A": 1, "C": 3.333333333333333333333333, "B": 2}),
        PythonNodeMap({"A": 1, "C": 3.333333333333333333333334, "B": 2 + 1e-9}),
    )
    with pytest.raises(AssertionError):
        PythonNodeMap.Type.assert_equal(
            PythonNodeMap({"A": 1}), PythonNodeMap({"A": 1, "B": 2})
        )
    with pytest.raises(AssertionError):
        PythonNodeMap.Type.assert_equal(
            PythonNodeMap({"A": 1, "B": 22}), PythonNodeMap({"A": 1, "B": 2})
        )
    with pytest.raises(AssertionError):
        PythonNodeMap.Type.assert_equal(
            PythonNodeMap({"A": 1.1}), PythonNodeMap({"A": 1})
        )
    # Node index doesn't affect equality
    PythonNodeMap.Type.assert_equal(
        PythonNodeMap({"A": 1, "B": 2, "C": 3}, node_index=IndexedNodes("ABC")),
        PythonNodeMap({"A": 1, "B": 2, "C": 3}, node_index=IndexedNodes("CBA")),
    )
    with pytest.raises(AssertionError):
        PythonNodeMap.Type.assert_equal(5, 5)


def test_numpy():
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(np.array([1, 3, 5, 7, 9])), NumpyNodeMap(np.array([1, 3, 5, 7, 9]))
    )
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(np.array([1, 3, 5.5555555555555555555, 7, 9])),
        NumpyNodeMap(np.array([1, 3, 5.5555555555555555556, 7, 9 + 1e-9])),
    )
    with pytest.raises(AssertionError):
        NumpyNodeMap.Type.assert_equal(
            NumpyNodeMap(np.array([1, 3, 5, 7, 9])),
            NumpyNodeMap(np.array([1, 3, 5, 7, 9, 11])),
        )
    # weights don't match, so we take the fast path and declare them not equal
    with pytest.raises(AssertionError):
        NumpyNodeMap.Type.assert_equal(
            NumpyNodeMap(np.array([1, 3, 5, 7, 9])),
            NumpyNodeMap(np.array([1, 3, 5, 7, 9]), weights="any"),
        )
    # Missing value should not affect equality
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(
            np.array([1, -1, -1, 4, 5, -1]),
            missing_mask=np.array([False, True, True, False, False, True]),
        ),
        NumpyNodeMap(
            np.array([1, 0, 0, 4, 5, 0]),
            missing_mask=np.array([False, True, True, False, False, True]),
        ),
    )
    # Node index rearrangement
    NumpyNodeMap.Type.assert_equal(
        NumpyNodeMap(
            np.array([1, 4, 5, -1, 0, 1]),
            missing_mask=np.array([False, False, False, True, True, True]),
            node_index=IndexedNodes("ADEFBC"),
        ),
        NumpyNodeMap(
            np.array([1, -1, -1, 4, 5, -1]),
            missing_mask=np.array([False, True, True, False, False, True]),
            node_index=IndexedNodes("ABCDEF"),
        ),
    )
    with pytest.raises(AssertionError):
        NumpyNodeMap.Type.assert_equal(5, 5)


def test_numpy_compact():
    CompactNumpyNodeMap.Type.assert_equal(
        CompactNumpyNodeMap(np.array([1, 3, 5]), {0: 0, 240: 1, 968: 2}),
        CompactNumpyNodeMap(np.array([1, 3, 5]), {0: 0, 240: 1, 968: 2}),
    )
    CompactNumpyNodeMap.Type.assert_equal(
        CompactNumpyNodeMap(
            np.array([1, 3, 5.5555555555555555555]), {0: 0, 1: 1, 2: 2}
        ),
        CompactNumpyNodeMap(
            np.array([1, 3 + 1e-9, 5.5555555555555555556]), {0: 0, 1: 1, 2: 2}
        ),
    )
    with pytest.raises(AssertionError):
        CompactNumpyNodeMap.Type.assert_equal(
            CompactNumpyNodeMap(np.array([1, 3, 5]), {0: 0, 1: 1, 2: 2}),
            CompactNumpyNodeMap(np.array([1, 3, 5, 7]), {0: 0, 1: 1, 2: 2, 3: 3}),
        )
    # weights don't match, so we take the fast path and declare them not equal
    with pytest.raises(AssertionError):
        CompactNumpyNodeMap.Type.assert_equal(
            CompactNumpyNodeMap(np.array([1, 3, 5]), {0: 0, 1: 1, 2: 2}),
            CompactNumpyNodeMap(np.array([1, 3, 5]), {0: 0, 1: 1, 2: 2}, weights="any"),
        )
    # Storage reorder
    CompactNumpyNodeMap.Type.assert_equal(
        CompactNumpyNodeMap(np.array([1, 3, 5]), {"A": 0, "B": 1, "C": 2}),
        CompactNumpyNodeMap(np.array([5, 1, 3]), {"C": 0, "A": 1, "B": 2}),
    )
    # Node index doesn't affect equality
    CompactNumpyNodeMap.Type.assert_equal(
        CompactNumpyNodeMap(
            np.array([1, 3, 5]),
            {"A": 0, "B": 1, "C": 2},
            node_index=IndexedNodes("ADEFBC"),
        ),
        CompactNumpyNodeMap(
            np.array([1, 3, 5]),
            {"A": 0, "B": 1, "C": 2},
            node_index=IndexedNodes("ABCDEF"),
        ),
    )
    with pytest.raises(AssertionError):
        CompactNumpyNodeMap.Type.assert_equal(5, 5)


def test_graphblas():
    GrblasNodeMap.Type.assert_equal(
        GrblasNodeMap(Vector.from_values([0, 1, 3, 4], [1, 2, 3, 4])),
        GrblasNodeMap(Vector.from_values([0, 1, 3, 4], [1, 2, 3, 4])),
    )
    GrblasNodeMap.Type.assert_equal(
        GrblasNodeMap(
            Vector.from_values([0, 1, 3, 4], [1.0, 2.0, 3.333333333333333333, 4.0])
        ),
        GrblasNodeMap(
            Vector.from_values([0, 1, 3, 4], [1.0, 2.0, 3.333333333333333334, 4 + 1e-9])
        ),
    )
    with pytest.raises(AssertionError):
        GrblasNodeMap.Type.assert_equal(
            GrblasNodeMap(Vector.from_values([0, 1, 3, 4], [1, 2, 3, 4])),
            GrblasNodeMap(Vector.from_values([0, 1, 2, 4], [1, 2, 3, 4])),
        )
    with pytest.raises(AssertionError):
        GrblasNodeMap.Type.assert_equal(
            GrblasNodeMap(Vector.from_values([0, 1, 2], [1, 2, 3])),
            GrblasNodeMap(Vector.from_values([0, 1, 2, 3], [1, 2, 3, 4])),
        )
    # weights don't match, so we take the fast path and declare them not equal
    with pytest.raises(AssertionError):
        GrblasNodeMap.Type.assert_equal(
            GrblasNodeMap(Vector.from_values([0, 1], [1, 2])),
            GrblasNodeMap(Vector.from_values([0, 1], [1, 2]), weights="any"),
        )
    # Node index affects comparison
    GrblasNodeMap.Type.assert_equal(
        GrblasNodeMap(
            Vector.from_values([0, 1, 4], [1, 2, 3], size=5),
            node_index=IndexedNodes("ABCDE"),
        ),
        GrblasNodeMap(
            Vector.from_values([0, 2, 3], [2, 3, 1], size=5),
            node_index=IndexedNodes("BDEAC"),
        ),
    )
    with pytest.raises(AssertionError):
        GrblasNodeMap.Type.assert_equal(5, 5)
