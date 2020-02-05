from metagraph import dataobj


def test_python_sparse_array():
    sparse_dict = {1: 1.1, 5: 5.5, 2: 2.2}
    x = dataobj.PythonSparseArray(sparse_dict, size=30)
    assert len(x) == 30
    # Convert back and forth from numpy dense array
    npd = x.to(dataobj.NumpyDenseArray)
    assert isinstance(npd, dataobj.NumpyDenseArray)
    y = npd.to(type(x))
    assert y.obj == sparse_dict
    assert len(y) == 30
    # Convert back and forth from grblas vector
    grbv = x.to(dataobj.GrblasVector)
    assert isinstance(grbv, dataobj.GrblasVector)
    z = grbv.to(type(x))
    assert z.obj == sparse_dict
    assert len(z) == 30


def test_numpy_dense_array():
    import numpy as np

    dense_array = np.array([0, 1.1, 0, 0, 4.4, 5.5, 6.6, 0])
    x = dataobj.NumpyDenseArray(dense_array, missing_value=0)
    assert len(x) == 8
    # Convert back and forth from python sparse array
    pysa = x.to(dataobj.PythonSparseArray)
    assert isinstance(pysa, dataobj.PythonSparseArray)
    assert len(pysa) == 8
    y = pysa.to(type(x))
    np.testing.assert_equal(
        y.obj, np.where(dense_array == 0, y.missing_value, dense_array)
    )
    # Convert back and forth from grblas vector
    grbv = x.to(dataobj.GrblasVector)
    assert isinstance(grbv, dataobj.GrblasVector)
    assert len(grbv) == 8
    z = grbv.to(type(x))
    np.testing.assert_equal(
        z.obj, np.where(dense_array == 0, z.missing_value, dense_array)
    )


def test_grblas_vector():
    import grblas

    vec = grblas.Vector.new_from_values([0, 4, 9], [0.0, 4.4, 9.9], size=15)
    x = dataobj.GrblasVector(vec)
    assert len(x) == 15
    # Convert back and forth from python sparse array
    pysa = x.to(dataobj.PythonSparseArray)
    assert isinstance(pysa, dataobj.PythonSparseArray)
    assert len(pysa) == 15
    y = pysa.to(type(x))
    assert y.obj == vec
    # Convert back and forth from numpy dense array
    npd = x.to(dataobj.NumpyDenseArray)
    assert isinstance(npd, dataobj.NumpyDenseArray)
    assert len(npd) == 15
    z = npd.to(type(x))
    assert z.obj == vec
