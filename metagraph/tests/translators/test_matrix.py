import pytest
from metagraph.plugins.numpy.types import NumpyMatrix
from metagraph.plugins.graphblas.types import GrblasMatrixType
from metagraph.plugins.scipy.types import ScipyMatrixType
import numpy as np
import grblas
import scipy.sparse as ss


def test_numpy(default_plugin_resolver):
    dpr = default_plugin_resolver
    mat = np.array([[1, 2, 3], [3, 3, 9], [3, 0, 3]])
    x = NumpyMatrix(mat, missing_value=3)
    assert len(x.value[~x.get_missing_mask()]) == 4
    assert x.shape == (3, 3)
    # Convert back and forth from scipy.sparse
    ssm = dpr.translate(x, ss.spmatrix)
    assert isinstance(ssm, ss.spmatrix)
    y = dpr.translate(ssm, NumpyMatrix)
    assert isinstance(y, NumpyMatrix)
    np.testing.assert_equal(y.value, np.where(mat == 3, y.missing_value, mat))


def test_grblas(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = grblas.Matrix.new_from_values(
        [0, 0, 1, 2],
        [0, 1, 2, 1],
        [1, 2, 9, 0],
        nrows=3,
        ncols=4,
        dtype=grblas.dtypes.FP64,
    )
    assert x.nvals == 4
    assert x.shape == (3, 4)
    # Convert back and forth from scipy.sparse
    ssm = dpr.translate(x, ss.spmatrix)
    assert isinstance(ssm, ss.spmatrix)
    y = dpr.translate(ssm, grblas.Matrix)
    assert isinstance(y, grblas.Matrix)
    assert x == y


def test_scipy(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = ss.coo_matrix(([2.2, 1.1, -1.0], ([1, 1, 2], [0, 2, 1])), shape=(3, 3))
    assert x.nnz == 3
    # Convert back and forth from numpy
    npm = dpr.translate(x, NumpyMatrix)
    assert isinstance(npm, NumpyMatrix)
    y = dpr.translate(npm, ScipyMatrixType)
    assert isinstance(y, ss.spmatrix)
    assert (x != y).nnz == 0
    # Convert back and forth from grblas
    gbm = dpr.translate(x, GrblasMatrixType)
    assert isinstance(gbm, grblas.Matrix)
    z = dpr.translate(gbm, ScipyMatrixType)
    assert isinstance(z, ss.spmatrix)
    assert (x != z).nnz == 0

    a = ss.csr_matrix([[1, 0], [0, 0]])  # np.longlong by default
    b = dpr.translate(a, GrblasMatrixType)
    assert isinstance(b, grblas.Matrix)
    c = dpr.translate(b, ScipyMatrixType)
    assert (a != c).nnz == 0
    assert a.dtype == c.dtype
