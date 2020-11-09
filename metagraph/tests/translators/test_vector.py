import pytest

grblas = pytest.importorskip("grblas")

from metagraph.tests.util import default_plugin_resolver
from . import RoundTripper
from metagraph.plugins.numpy.types import NumpyVectorType
from metagraph.plugins.graphblas.types import GrblasVectorType
import numpy as np


def test_vector_roundtrip(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    dense_array = np.array([-4.2, 1.1, 0.0, 0.0, 4.4, 5.5, 6.6, 16.67])
    rt.verify_round_trip(dense_array)
    rt.verify_round_trip(dense_array.astype(int))
    rt.verify_round_trip(dense_array.astype(bool))


def test_numpy_2_graphblas(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = np.array([0, 1.1, 0, 0, 4.4, 5.5, 6.6, 0])
    assert len(x) == 8
    # Convert numpy -> grblas vector
    intermediate = grblas.Vector.from_values(
        [0, 1, 2, 3, 4, 5, 6, 7], [0.0, 1.1, 0.0, 0.0, 4.4, 5.5, 6.6, 0.0], size=8
    )
    y = dpr.translate(x, GrblasVectorType)
    dpr.assert_equal(y, intermediate)
    # Convert numpy <- grblas vector
    x2 = dpr.translate(y, NumpyVectorType)
    dpr.assert_equal(x, x2)
