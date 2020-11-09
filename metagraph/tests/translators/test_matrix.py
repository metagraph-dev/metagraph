import pytest

grblas = pytest.importorskip("grblas")

from metagraph.tests.util import default_plugin_resolver
from . import RoundTripper
from metagraph.plugins.numpy.types import NumpyMatrixType
from metagraph.plugins.graphblas.types import GrblasMatrixType
import numpy as np


def test_matrix_roundtrip_dense_square(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    mat = np.array([[1.1, 2.2, 3.3], [3.3, 3.3, 9.9], [3.3, 0.0, -3.3]])
    rt.verify_round_trip(mat)
    rt.verify_round_trip(mat.astype(int))
    rt.verify_round_trip(mat.astype(bool))


def test_matrix_roundtrip_dense_rect(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    mat = np.array(
        [[1.1, 2.2, 3.3], [3.3, 3.3, 9.9], [3.3, 0.0, -3.3], [-1.1, 2.7, 3.3]]
    )
    rt.verify_round_trip(mat)
    rt.verify_round_trip(mat.astype(int))
    rt.verify_round_trip(mat.astype(bool))


def test_numpy_2_grblas(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = np.array([[1, 2, 3], [3, 3, 9], [3, 0, 3], [4, 2, 2]])
    assert x.shape == (4, 3)
    # Convert numpy -> grblas.Matrix
    intermediate = grblas.Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        [1, 2, 3, 3, 3, 9, 3, 0, 3, 4, 2, 2],
        nrows=4,
        ncols=3,
        dtype=grblas.dtypes.INT64,
    )
    y = dpr.translate(x, grblas.Matrix)
    dpr.assert_equal(y, intermediate)
    # Convert numpy <- grblas.Matrix
    x2 = dpr.translate(y, NumpyMatrixType)
    dpr.assert_equal(x, x2)
