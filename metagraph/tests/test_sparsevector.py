import pytest
from metagraph.default_plugins.wrappers.python import PythonSparseVector
from metagraph.default_plugins.wrappers.numpy import NumpySparseVector
from metagraph.default_plugins.wrappers.graphblas import GrblasVector
import grblas


def test_python():
    pytest.xfail("Needs to be updated to use resolver")
    sparse_dict = {1: 1.1, 5: 5.5, 2: 2.2}
    x = PythonSparseVector(sparse_dict, size=30)
    assert len(x) == 30
    # Convert back and forth from numpy dense array
    npd = x.to(NumpySparseVector)
    assert isinstance(npd, NumpySparseVector)
    y = npd.to(PythonSparseVector)
    assert y.obj == sparse_dict
    assert len(y) == 30
    # Convert back and forth from grblas vector
    grbv = x.to(GrblasVector)
    assert isinstance(grbv, grblas.Vector)
    z = grbv.to(PythonSparseVector)
    assert z.obj == sparse_dict
    assert len(z) == 30


def test_numpy():
    pytest.xfail("Needs to be updated to use resolver")
    import numpy as np

    dense_array = np.array([0, 1.1, 0, 0, 4.4, 5.5, 6.6, 0])
    x = NumpySparseVector(dense_array, missing_value=0)
    assert len(x) == 8
    # Convert back and forth from python sparse array
    pysa = x.to(PythonSparseVector)
    assert isinstance(pysa, PythonSparseVector)
    assert len(pysa) == 8
    y = pysa.to(NumpySparseVector)
    np.testing.assert_equal(
        y.obj, np.where(dense_array == 0, y.missing_value, dense_array)
    )
    # Convert back and forth from grblas vector
    grbv = x.to(GrblasVector)
    assert isinstance(grbv, grblas.Vector)
    assert len(grbv) == 8
    z = grbv.to(NumpySparseVector)
    np.testing.assert_equal(
        z.obj, np.where(dense_array == 0, z.missing_value, dense_array)
    )


def test_grblas():
    pytest.xfail("Needs to be updated to use resolver")
    import grblas

    x = grblas.Vector.new_from_values([0, 4, 9], [0.0, 4.4, 9.9], size=15)
    assert len(x) == 15
    # Convert back and forth from python sparse array
    pysa = x.to(PythonSparseVector)
    assert isinstance(pysa, PythonSparseVector)
    assert len(pysa) == 15
    y = pysa.to(GrblasVector)
    assert y.obj == x
    # Convert back and forth from numpy dense array
    npd = x.to(NumpySparseVector)
    assert isinstance(npd, NumpySparseVector)
    assert len(npd) == 15
    z = npd.to(GrblasVector)
    assert z.obj == x
