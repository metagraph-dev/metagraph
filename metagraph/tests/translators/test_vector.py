import pytest

grblas = pytest.importorskip("grblas")

from metagraph.tests.util import default_plugin_resolver
from metagraph.plugins.numpy.types import NumpyVector
from metagraph.plugins.graphblas.types import GrblasVectorType
import numpy as np


def test_numpy_2_graphblas(default_plugin_resolver):
    dpr = default_plugin_resolver
    dense_array = np.array([0, 1.1, 0, 0, 4.4, 5.5, 6.6, 0])
    missing_mask = dense_array == 0
    x = NumpyVector(dense_array, mask=~missing_mask)
    assert len(x) == 8
    # Convert numpy -> grblas vector
    intermediate = grblas.Vector.from_values([1, 4, 5, 6], [1.1, 4.4, 5.5, 6.6], size=8)
    y = dpr.translate(x, GrblasVectorType)
    dpr.assert_equal(y, intermediate)
    # Convert numpy <- grblas vector
    x2 = dpr.translate(y, NumpyVector)
    dpr.assert_equal(x, x2)
