import pytest
from metagraph.plugins.numpy.types import NumpyMatrix
from metagraph.plugins.graphblas.types import GrblasMatrixType
from metagraph.plugins.scipy.types import ScipyMatrixType
import numpy as np
import grblas
import scipy.sparse as ss


def test_numpy_2_scipy(default_plugin_resolver):
    dpr = default_plugin_resolver
    mat = np.array([[1, 2, 3], [3, 3, 9], [3, 0, 3]])
    missing_mask = mat == 3
    x = NumpyMatrix(mat, missing_mask=missing_mask)
    assert x.shape == (3, 3)
    # Convert numpy -> scipy.sparse
    intermediate = ss.coo_matrix(
        ([1, 2, 9, 0], ([0, 0, 1, 2], [0, 1, 2, 1])), shape=(3, 3)
    )
    y = dpr.translate(x, ss.spmatrix)
    assert ScipyMatrixType.compare_objects(y, intermediate)
    # Convert numpy <- scipy.sparse
    x2 = dpr.translate(y, NumpyMatrix)
    assert NumpyMatrix.Type.compare_objects(x, x2)


def test_grblas_2_scipy(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = grblas.Matrix.from_values(
        [0, 0, 1, 2],
        [0, 1, 2, 1],
        [1, 2, 9, 0],
        nrows=3,
        ncols=4,
        dtype=grblas.dtypes.FP64,
    )
    assert x.nvals == 4
    assert x.shape == (3, 4)
    # Convert grblas matrix -> scipy.sparse
    intermediate = ss.coo_matrix(
        ([1, 2, 9, 0], ([0, 0, 1, 2], [0, 1, 2, 1])), shape=(3, 4), dtype=np.float64
    )
    y = dpr.translate(x, ss.spmatrix)
    assert ScipyMatrixType.compare_objects(y, intermediate)
    # Convert grblas matrix <- scipy.sparse
    x2 = dpr.translate(y, grblas.Matrix)
    assert GrblasMatrixType.compare_objects(x, x2)
