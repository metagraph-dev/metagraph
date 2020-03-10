from metagraph.plugins.python.types import PythonSparseVector
from metagraph.plugins.numpy.types import NumpySparseVector
from metagraph.plugins.graphblas.types import GrblasVectorType
import grblas


def test_python(default_plugin_resolver):
    dpr = default_plugin_resolver
    sparse_dict = {1: 1.1, 5: 5.5, 2: 2.2}
    x = PythonSparseVector(sparse_dict, size=30)
    assert len(x) == 30
    # Convert back and forth from numpy sparse array
    nps = dpr.translate(x, NumpySparseVector)
    assert isinstance(nps, NumpySparseVector)
    y = dpr.translate(nps, PythonSparseVector)
    assert y.value == sparse_dict
    assert len(y) == 30
    # Convert back and forth from grblas vector
    grbv = dpr.translate(x, GrblasVectorType)
    assert isinstance(grbv, grblas.Vector)
    z = dpr.translate(grbv, PythonSparseVector)
    assert z.value == sparse_dict
    assert len(z) == 30


def test_numpy(default_plugin_resolver):
    import numpy as np

    dpr = default_plugin_resolver
    dense_array = np.array([0, 1.1, 0, 0, 4.4, 5.5, 6.6, 0])
    x = NumpySparseVector(dense_array, missing_value=0)
    assert len(x) == 8
    # Convert back and forth from python sparse array
    pysa = dpr.translate(x, PythonSparseVector)
    assert isinstance(pysa, PythonSparseVector)
    assert len(pysa) == 8
    y = dpr.translate(pysa, NumpySparseVector)
    np.testing.assert_equal(
        y.value, np.where(dense_array == 0, y.missing_value, dense_array)
    )
    # Convert back and forth from grblas vector
    grbv = dpr.translate(x, GrblasVectorType)
    assert isinstance(grbv, grblas.Vector)
    assert grbv.size == 8
    z = dpr.translate(grbv, NumpySparseVector)
    np.testing.assert_equal(
        z.value, np.where(dense_array == 0, z.missing_value, dense_array)
    )


def test_grblas(default_plugin_resolver):
    import grblas

    dpr = default_plugin_resolver
    x = grblas.Vector.new_from_values([0, 4, 9], [0.0, 4.4, 9.9], size=15)
    assert x.size == 15
    # Convert back and forth from python sparse array
    pysa = dpr.translate(x, PythonSparseVector)
    assert isinstance(pysa, PythonSparseVector)
    assert len(pysa) == 15
    y = dpr.translate(pysa, grblas.Vector)
    assert y == x
    # Convert back and forth from numpy sparse array
    nps = dpr.translate(x, NumpySparseVector)
    assert isinstance(nps, NumpySparseVector)
    assert len(nps) == 15
    z = dpr.translate(nps, grblas.Vector)
    assert z == x
