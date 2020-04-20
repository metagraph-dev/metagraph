import pytest
from metagraph.plugins.numpy.types import NumpyMatrix
from metagraph.plugins.scipy.types import ScipyMatrixType
from metagraph.plugins.graphblas.types import GrblasMatrixType
import numpy as np
import scipy.sparse as ss
import grblas


def test_numpy():
    with pytest.raises(TypeError):
        NumpyMatrix(np.array([1, 2, 3]))
    assert NumpyMatrix.Type.compare_objects(
        NumpyMatrix(np.array([[1, 2, 3], [4, 5, 6]])),
        NumpyMatrix(np.array([[1, 2, 3], [4, 5, 6]])),
    )
    assert NumpyMatrix.Type.compare_objects(
        NumpyMatrix(np.array([[1.1, 2.2, 3.333333333333333]])),
        NumpyMatrix(np.array([[1.1, 2.199999999999999, 3.3333333333333334]])),
    )
    # Different size
    assert not NumpyMatrix.Type.compare_objects(
        NumpyMatrix(np.array([[1, 2, 3], [4, 5, 6]])),
        NumpyMatrix(np.array([[1, 2], [3, 4], [5, 6]])),
    )
    # Different dtypes are not equal
    assert not NumpyMatrix.Type.compare_objects(
        NumpyMatrix(np.array([[1, 2, 3]], dtype=np.int16)),
        NumpyMatrix(np.array([[1, 2, 3]], dtype=np.int32)),
    )
    # Missing values are ignored
    assert NumpyMatrix.Type.compare_objects(
        NumpyMatrix(
            np.array([[1, 2, 3], [4, 5, 6]]),
            missing_mask=np.array([[False, True, False], [False, False, True]]),
        ),
        NumpyMatrix(
            np.array([[1, 2, 3], [4, 5, 22]]),
            missing_mask=np.array([[False, True, False], [False, False, True]]),
        ),
    )
    # Different missing values
    assert not NumpyMatrix.Type.compare_objects(
        NumpyMatrix(
            np.array([[1, 2, 3, 4]]),
            missing_mask=np.array([[False, False, True, True]]),
        ),
        NumpyMatrix(
            np.array([[1, 2, 3, 4]]),
            missing_mask=np.array([[False, False, False, True]]),
        ),
    )
    # Coincidental equality of non-missing values is not equality
    assert not NumpyMatrix.Type.compare_objects(
        NumpyMatrix(
            np.array([[1, 2, 3], [3, 3, 3]]),
            missing_mask=np.array([[False, False, True], [False, False, False]]),
        ),
        NumpyMatrix(
            np.array([[1, 2, 3], [3, 3, 3]]),
            missing_mask=np.array([[False, False, False], [True, False, False]]),
        ),
    )
    with pytest.raises(TypeError):
        NumpyMatrix.Type.compare_objects(5, 5)


def test_scipy():
    assert ScipyMatrixType.compare_objects(
        ss.coo_matrix(np.array([[1, 2, 3], [4, 5, 6]])),
        ss.coo_matrix(np.array([[1, 2, 3], [4, 5, 6]])),
    )
    assert ScipyMatrixType.compare_objects(
        ss.coo_matrix(np.array([[1.1, 2.2, 3.333333333333333]])),
        ss.coo_matrix(np.array([[1.1, 2.199999999999999, 3.3333333333333334]])),
    )
    # Different size
    assert not ScipyMatrixType.compare_objects(
        ss.coo_matrix(np.array([[1, 2, 3], [4, 5, 6]])),
        ss.coo_matrix(np.array([[1, 2], [3, 4], [5, 6]])),
    )
    # Different dtypes are not equal
    assert not ScipyMatrixType.compare_objects(
        ss.coo_matrix(np.array([[1, 2, 3]], dtype=np.int16)),
        ss.coo_matrix(np.array([[1, 2, 3]], dtype=np.int32)),
    )
    # Missing values are ignored
    assert ScipyMatrixType.compare_objects(
        ss.coo_matrix(([1, 3, 4, 5], ([0, 0, 1, 1], [0, 2, 0, 1]))),
        ss.coo_matrix(([1, 3, 4, 5], ([0, 0, 1, 1], [0, 2, 0, 1]))),
    )
    # Different missing values
    assert not ScipyMatrixType.compare_objects(
        ss.coo_matrix(([1, 3, 4, 5], ([0, 0, 1, 1], [0, 2, 0, 1]))),
        ss.coo_matrix(([1, 3, 4, 5], ([0, 0, 1, 1], [0, 2, 0, 2]))),
    )
    with pytest.raises(TypeError):
        ScipyMatrixType.compare_objects(5, 5)


def test_graphblas():
    assert GrblasMatrixType.compare_objects(
        grblas.Matrix.new_from_values(
            [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [1, 2, 3, 4, 5, 6]
        ),
        grblas.Matrix.new_from_values(
            [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [1, 2, 3, 4, 5, 6]
        ),
    )
    assert GrblasMatrixType.compare_objects(
        grblas.Matrix.new_from_values(
            [0, 0, 0], [0, 1, 2], [1.1, 2.2, 3.33333333333333333]
        ),
        grblas.Matrix.new_from_values(
            [0, 0, 0], [0, 1, 2], [1.1, 2.19999999999999999, 3.333333333333333334]
        ),
    )
    # Different size
    assert not GrblasMatrixType.compare_objects(
        grblas.Matrix.new_from_values(
            [0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [1, 2, 3, 4, 5, 6]
        ),
        grblas.Matrix.new_from_values(
            [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], [1, 2, 3, 4, 5, 6]
        ),
    )
    # Different dtypes are not equal
    assert not GrblasMatrixType.compare_objects(
        grblas.Matrix.new_from_values(
            [0, 0, 0], [0, 1, 2], [1, 2, 3], dtype=grblas.dtypes.INT16
        ),
        grblas.Matrix.new_from_values(
            [0, 0, 0], [0, 1, 2], [1, 2, 3], dtype=grblas.dtypes.INT32
        ),
    )
    # Missing values are ignored
    assert GrblasMatrixType.compare_objects(
        grblas.Matrix.new_from_values([0, 0, 1, 1], [0, 2, 0, 1], [1, 3, 4, 5]),
        grblas.Matrix.new_from_values([0, 0, 1, 1], [0, 2, 0, 1], [1, 3, 4, 5]),
    )
    # Different missing values
    assert not GrblasMatrixType.compare_objects(
        grblas.Matrix.new_from_values([0, 0, 1, 1], [0, 2, 0, 1], [1, 3, 4, 5]),
        grblas.Matrix.new_from_values([0, 0, 1, 1], [0, 2, 0, 2], [1, 3, 4, 5]),
    )
    with pytest.raises(TypeError):
        GrblasMatrixType.compare_objects(5, 5)
