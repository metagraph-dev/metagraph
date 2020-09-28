import pytest

grblas = pytest.importorskip("grblas")

from metagraph.plugins.numpy.types import NumpyMatrix
from metagraph.plugins.scipy.types import ScipyMatrixType
from metagraph.plugins.graphblas.types import GrblasMatrixType
import numpy as np
import scipy.sparse as ss


def test_numpy():
    with pytest.raises(TypeError):
        NumpyMatrix(np.array([1, 2, 3]))
    NumpyMatrix.Type.assert_equal(
        NumpyMatrix(np.array([[1, 2, 3], [4, 5, 6]])),
        NumpyMatrix(np.array([[1, 2, 3], [4, 5, 6]])),
        {},
        {},
        {},
        {},
    )
    NumpyMatrix.Type.assert_equal(
        NumpyMatrix(np.array([[1.1, 2.2, 3.333333333333333]])),
        NumpyMatrix(np.array([[1.1, 2.199999999999999, 3.3333333333333334]])),
        {},
        {},
        {},
        {},
    )
    # Different size
    with pytest.raises(AssertionError):
        NumpyMatrix.Type.assert_equal(
            NumpyMatrix(np.array([[1, 2, 3], [4, 5, 6]])),
            NumpyMatrix(np.array([[1, 2], [3, 4], [5, 6]])),
            {},
            {},
            {},
            {},
        )
    # Different dtypes are not equal
    with pytest.raises(AssertionError):
        NumpyMatrix.Type.assert_equal(
            NumpyMatrix(np.array([[1, 2, 3]], dtype=np.int16)),
            NumpyMatrix(np.array([[1, 2, 3]], dtype=np.int32)),
            {"dtype": "int"},
            {"dtype": "float"},
            {},
            {},
        )
    # Missing values are ignored
    NumpyMatrix.Type.assert_equal(
        NumpyMatrix(
            np.array([[1, 2, 3], [4, 5, 6]]),
            mask=np.array([[True, False, True], [True, True, False]]),
        ),
        NumpyMatrix(
            np.array([[1, 2, 3], [4, 5, 22]]),
            mask=np.array([[True, False, True], [True, True, False]]),
        ),
        {},
        {},
        {},
        {},
    )
    # Different missing values
    with pytest.raises(AssertionError):
        NumpyMatrix.Type.assert_equal(
            NumpyMatrix(
                np.array([[1, 2, 3, 4]]), mask=np.array([[False, False, True, True]]),
            ),
            NumpyMatrix(
                np.array([[1, 2, 3, 4]]), mask=np.array([[False, False, False, True]]),
            ),
            {},
            {},
            {},
            {},
        )
    # Coincidental equality of non-missing values is not equality
    with pytest.raises(AssertionError):
        NumpyMatrix.Type.assert_equal(
            NumpyMatrix(
                np.array([[1, 2, 3], [3, 3, 3]]),
                mask=np.array([[True, True, False], [True, True, True]]),
            ),
            NumpyMatrix(
                np.array([[1, 2, 3], [3, 3, 3]]),
                mask=np.array([[True, True, True], [False, True, True]]),
            ),
            {},
            {},
            {},
            {},
        )


def test_scipy():
    ScipyMatrixType.assert_equal(
        ss.coo_matrix(np.array([[1, 2, 3], [4, 5, 6]])),
        ss.coo_matrix(np.array([[1, 2, 3], [4, 5, 6]])),
        {},
        {},
        {},
        {},
    )
    ScipyMatrixType.assert_equal(
        ss.coo_matrix(np.array([[1.1, 2.2, 3.333333333333333]])),
        ss.coo_matrix(np.array([[1.1, 2.199999999999999, 3.3333333333333334]])),
        {},
        {},
        {},
        {},
    )
    # Different size
    with pytest.raises(AssertionError):
        ScipyMatrixType.assert_equal(
            ss.coo_matrix(np.array([[1, 2, 3], [4, 5, 6]])),
            ss.coo_matrix(np.array([[1, 2], [3, 4], [5, 6]])),
            {},
            {},
            {},
            {},
        )
    # Different dtypes are not equal
    with pytest.raises(AssertionError):
        ScipyMatrixType.assert_equal(
            ss.coo_matrix(np.array([[1, 2, 3]], dtype=np.int16)),
            ss.coo_matrix(np.array([[1, 2, 3]], dtype=np.int32)),
            {"dtype": "int"},
            {"dtype": "float"},
            {},
            {},
        )
    # Missing values are ignored
    ScipyMatrixType.assert_equal(
        ss.coo_matrix(([1, 3, 4, 5], ([0, 0, 1, 1], [0, 2, 0, 1]))),
        ss.coo_matrix(([1, 3, 4, 5], ([0, 0, 1, 1], [0, 2, 0, 1]))),
        {},
        {},
        {},
        {},
    )
    # Different missing values
    with pytest.raises(AssertionError):
        ScipyMatrixType.assert_equal(
            ss.coo_matrix(([1, 3, 4, 5], ([0, 0, 1, 1], [0, 2, 0, 1]))),
            ss.coo_matrix(([1, 3, 4, 5], ([0, 0, 1, 1], [0, 2, 0, 2]))),
            {},
            {},
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
    # Missing values are ignored
    GrblasMatrixType.assert_equal(
        grblas.Matrix.from_values([0, 0, 1, 1], [0, 2, 0, 1], [1, 3, 4, 5]),
        grblas.Matrix.from_values([0, 0, 1, 1], [0, 2, 0, 1], [1, 3, 4, 5]),
        {},
        {},
        {},
        {},
    )
    # Different missing values
    with pytest.raises(AssertionError):
        GrblasMatrixType.assert_equal(
            grblas.Matrix.from_values([0, 0, 1, 1], [0, 2, 0, 1], [1, 3, 4, 5]),
            grblas.Matrix.from_values([0, 0, 1, 1], [0, 2, 0, 2], [1, 3, 4, 5]),
            {},
            {},
            {},
            {},
        )
