import pytest

grblas = pytest.importorskip("grblas")

from metagraph.plugins.numpy.types import NumpyMatrixType
from metagraph.plugins.graphblas.types import GrblasMatrixType
import numpy as np


def test_numpy():
    NumpyMatrixType.assert_equal(
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[1, 2, 3], [4, 5, 6]]),
        {},
        {},
        {},
        {},
    )
    NumpyMatrixType.assert_equal(
        np.array([[1.1, 2.2, 3.333333333333333]]),
        np.array([[1.1, 2.199999999999999, 3.3333333333333334]]),
        {},
        {},
        {},
        {},
    )
    # Different size
    with pytest.raises(AssertionError):
        NumpyMatrixType.assert_equal(
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            {},
            {},
            {},
            {},
        )
    # Different dtypes are not equal
    with pytest.raises(AssertionError):
        NumpyMatrixType.assert_equal(
            np.array([[1, 2, 3]], dtype=np.int16),
            np.array([[1, 2, 3]], dtype=np.int32),
            {"dtype": "int"},
            {"dtype": "float"},
            {},
            {},
        )


def test_graphblas():
    GrblasMatrixType.assert_equal(
        grblas.Matrix.from_values(
            [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [1, 2, 3, 4, 5, 6]
        ),
        grblas.Matrix.from_values(
            [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [1, 2, 3, 4, 5, 6]
        ),
        {},
        {},
        {},
        {},
    )
    GrblasMatrixType.assert_equal(
        grblas.Matrix.from_values(
            [0, 0, 0], [0, 1, 2], [1.1, 2.2, 3.33333333333333333]
        ),
        grblas.Matrix.from_values(
            [0, 0, 0], [0, 1, 2], [1.1, 2.19999999999999999, 3.333333333333333334]
        ),
        {},
        {},
        {},
        {},
    )
    # Different size
    with pytest.raises(AssertionError):
        GrblasMatrixType.assert_equal(
            grblas.Matrix.from_values(
                [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [1, 2, 3, 4, 5, 6]
            ),
            grblas.Matrix.from_values(
                [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], [1, 2, 3, 4, 5, 6]
            ),
            {},
            {},
            {},
            {},
        )
    # Different dtypes are not equal
    with pytest.raises(AssertionError):
        GrblasMatrixType.assert_equal(
            grblas.Matrix.from_values(
                [0, 0, 0], [0, 1, 2], [1, 2, 3], dtype=grblas.dtypes.INT16
            ),
            grblas.Matrix.from_values(
                [0, 0, 0], [0, 1, 2], [1, 2, 3], dtype=grblas.dtypes.INT32
            ),
            {},
            {},
            {},
            {},
        )
