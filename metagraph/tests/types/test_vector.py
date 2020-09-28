import pytest

grblas = pytest.importorskip("grblas")

from metagraph.plugins.numpy.types import NumpyVector
from metagraph.plugins.graphblas.types import GrblasVectorType
import numpy as np


def test_numpy():
    with pytest.raises(TypeError):
        NumpyVector(np.array([[1, 2, 3], [4, 5, 6]]))
    NumpyVector.Type.assert_equal(
        NumpyVector(np.array([1, 2, 3])),
        NumpyVector(np.array([1, 2, 3])),
        {},
        {},
        {},
        {},
    )
    NumpyVector.Type.assert_equal(
        NumpyVector(np.array([1.1, 2.2, 3.333333333333333])),
        NumpyVector(np.array([1.1, 2.199999999999999, 3.3333333333333334])),
        {},
        {},
        {},
        {},
    )
    # Different size
    with pytest.raises(AssertionError):
        NumpyVector.Type.assert_equal(
            NumpyVector(np.array([1, 2, 3])),
            NumpyVector(np.array([1, 2, 3, 4])),
            {},
            {},
            {},
            {},
        )
    # Different dtypes are not equal
    with pytest.raises(AssertionError):
        NumpyVector.Type.assert_equal(
            NumpyVector(np.array([1, 2, 3], dtype=np.int16)),
            NumpyVector(np.array([1, 2, 3], dtype=np.int32)),
            {"dtype": "int"},
            {"dtype": "float"},
            {},
            {},
        )
    # Missing values are ignored
    NumpyVector.Type.assert_equal(
        NumpyVector(np.array([1, 2, 3, 4]), mask=np.array([True, False, True, False])),
        NumpyVector(np.array([1, 2, 3, 22]), mask=np.array([True, False, True, False])),
        {"dtype": "bool", "is_dense": False},
        {"dtype": "bool", "is_dense": False},
        {},
        {},
    )
    # Different missing values
    with pytest.raises(AssertionError):
        NumpyVector.Type.assert_equal(
            NumpyVector(
                np.array([1, 2, 3, 4]), mask=np.array([False, False, True, True]),
            ),
            NumpyVector(
                np.array([1, 2, 3, 4]), mask=np.array([False, False, False, True]),
            ),
            {},
            {},
            {},
            {},
        )
    # Coincidental equality of non-missing values is not equality
    with pytest.raises(AssertionError):
        NumpyVector.Type.assert_equal(
            NumpyVector(
                np.array([1, 2, 3, 3]), mask=np.array([True, True, False, True]),
            ),
            NumpyVector(
                np.array([1, 2, 3, 3]), mask=np.array([True, True, True, False]),
            ),
            {},
            {},
            {},
            {},
        )


def test_graphblas():
    GrblasVectorType.assert_equal(
        grblas.Vector.from_values([0, 1, 2], [1, 2, 3]),
        grblas.Vector.from_values([0, 1, 2], [1, 2, 3]),
        {},
        {},
        {},
        {},
    )
    GrblasVectorType.assert_equal(
        grblas.Vector.from_values([0, 1, 2], [1.1, 2.2, 3.333333333333333333]),
        grblas.Vector.from_values(
            [0, 1, 2], [1.1, 2.19999999999999999, 3.333333333333333334]
        ),
        {},
        {},
        {},
        {},
    )
    # Different size
    with pytest.raises(AssertionError):
        GrblasVectorType.assert_equal(
            grblas.Vector.from_values([0, 1, 2], [1, 2, 3]),
            grblas.Vector.from_values([0, 1, 2, 3], [1, 2, 3, 4]),
            {},
            {},
            {},
            {},
        )
    # Different dtypes are not equal
    with pytest.raises(AssertionError):
        GrblasVectorType.assert_equal(
            grblas.Vector.from_values([0, 1, 2], [1, 2, 3], dtype=grblas.dtypes.INT16),
            grblas.Vector.from_values([0, 1, 2], [1, 2, 3], dtype=grblas.dtypes.INT32),
            {},
            {},
            {},
            {},
        )
    # Sparse vector
    GrblasVectorType.assert_equal(
        grblas.Vector.from_values([0, 2], [1, 3], size=4),
        grblas.Vector.from_values([0, 2], [1, 3], size=4),
        {},
        {},
        {},
        {},
    )
    # Different missing values
    with pytest.raises(AssertionError):
        GrblasVectorType.assert_equal(
            grblas.Vector.from_values([0, 1, 3], [1, 2, 4], size=4),
            grblas.Vector.from_values([0, 1, 2, 3], [1, 2, 3, 4], size=4),
            {},
            {},
            {},
            {},
        )
    # Coincidental equality of non-missing values is not equality
    with pytest.raises(AssertionError):
        GrblasVectorType.assert_equal(
            grblas.Vector.from_values([0, 2], [1, 3], size=4),
            grblas.Vector.from_values([0, 3], [1, 3], size=4),
            {},
            {},
            {},
            {},
        )
