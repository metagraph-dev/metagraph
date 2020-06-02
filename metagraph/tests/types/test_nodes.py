import pytest
from metagraph.plugins.python.types import PythonNodeMap
from metagraph.plugins.numpy.types import NumpyNodeMap, CompactNumpyNodeMap
from metagraph.plugins.graphblas.types import GrblasNodeMap
from metagraph import NodeLabels
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
    # # Storage reorder (this doesn't work -- should it?)
    # CompactNumpyNodeMap.Type.assert_equal(
    #     CompactNumpyNodeMap(np.array([1, 3, 5]), {0: 0, 2: 1, 7: 2}),
    #     CompactNumpyNodeMap(np.array([5, 1, 3]), {7: 0, 0: 1, 2: 2}),
    # )


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
