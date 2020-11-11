import pytest

grblas = pytest.importorskip("grblas")

from metagraph.plugins.numpy.types import NumpyVectorType
from metagraph.plugins.graphblas.types import GrblasVectorType
import numpy as np


def test_numpy():
    NumpyVectorType.assert_equal(
        np.array([1, 2, 3]), np.array([1, 2, 3]), {}, {}, {}, {}
    )
    NumpyVectorType.assert_equal(
        np.array([1.1, 2.2, 3.333333333333333]),
        np.array([1.1, 2.199999999999999, 3.3333333333333334]),
        {},
        {},
        {},
        {},
    )
    # Different size
    with pytest.raises(AssertionError):
        NumpyVectorType.assert_equal(
            np.array([1, 2, 3]), np.array([1, 2, 3, 4]), {}, {}, {}, {}
        )
    # Different dtypes are not equal
    with pytest.raises(AssertionError):
        NumpyVectorType.assert_equal(
            np.array([1, 2, 3], dtype=np.int16),
            np.array([1, 2, 3], dtype=np.int32),
            {"dtype": "int"},
            {"dtype": "float"},
            {},
            {},
        )


def test_graphblas():
    GrblasVectorType.assert_equal(
        grblas.Vector.from_values([0, 1, 2], [1, 2, 3]),
        grblas.Vector.from_values([0, 1, 2], [1, 2, 3]),
        {},
        {},
        {},
        {},
    )
    GrblasVectorType.assert_equal(
        grblas.Vector.from_values([0, 1, 2], [1.1, 2.2, 3.333333333333333333]),
        grblas.Vector.from_values(
            [0, 1, 2], [1.1, 2.19999999999999999, 3.333333333333333334]
        ),
        {},
        {},
        {},
        {},
    )
    # Different size
    with pytest.raises(AssertionError):
        GrblasVectorType.assert_equal(
            grblas.Vector.from_values([0, 1, 2], [1, 2, 3]),
            grblas.Vector.from_values([0, 1, 2, 3], [1, 2, 3, 4]),
            {},
            {},
            {},
            {},
        )
    # Different dtypes are not equal
    with pytest.raises(AssertionError):
        GrblasVectorType.assert_equal(
            grblas.Vector.from_values([0, 1, 2], [1, 2, 3], dtype=grblas.dtypes.INT16),
            grblas.Vector.from_values([0, 1, 2], [1, 2, 3], dtype=grblas.dtypes.INT32),
            {},
            {},
            {},
            {},
        )
