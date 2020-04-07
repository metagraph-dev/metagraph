from metagraph.plugins.numpy.types import NumpyVector
from metagraph.plugins.graphblas.types import GrblasVectorType
import grblas


def test_numpy(default_plugin_resolver):
    import numpy as np

    dpr = default_plugin_resolver
    dense_array = np.array([0, 1.1, 0, 0, 4.4, 5.5, 6.6, 0])
    x = NumpyVector(dense_array, missing_value=0)
    assert len(x) == 8
    # Convert back and forth from grblas vector
    grbv = dpr.translate(x, GrblasVectorType)
    assert isinstance(grbv, grblas.Vector)
    assert grbv.size == 8
    z = dpr.translate(grbv, NumpyVector)
    np.testing.assert_equal(
        z.value, np.where(dense_array == 0, z.missing_value, dense_array)
    )


def test_grblas(default_plugin_resolver):
    import grblas

    dpr = default_plugin_resolver
    x = grblas.Vector.new_from_values([0, 4, 9], [0.0, 4.4, 9.9], size=15)
    assert x.size == 15
    # Convert back and forth from numpy sparse array
    nps = dpr.translate(x, NumpyVector)
    assert isinstance(nps, NumpyVector)
    assert len(nps) == 15
    z = dpr.translate(nps, grblas.Vector)
    assert z == x
