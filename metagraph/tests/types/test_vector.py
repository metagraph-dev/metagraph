import pytest
from metagraph.plugins.numpy.types import NumpyVector
from metagraph.plugins.graphblas.types import GrblasVectorType
import numpy as np
import grblas


def test_numpy():
    with pytest.raises(TypeError):
        NumpyVector(np.array([[1, 2, 3], [4, 5, 6]]))
    assert NumpyVector.Type.compare_objects(
        NumpyVector(np.array([1, 2, 3])), NumpyVector(np.array([1, 2, 3]))
    )
    assert NumpyVector.Type.compare_objects(
        NumpyVector(np.array([1.1, 2.2, 3.333333333333333])),
        NumpyVector(np.array([1.1, 2.199999999999999, 3.3333333333333334])),
    )
    # Different size
    assert not NumpyVector.Type.compare_objects(
        NumpyVector(np.array([1, 2, 3])), NumpyVector(np.array([1, 2, 3, 4]))
    )
    # Different dtypes are not equal
    assert not NumpyVector.Type.compare_objects(
        NumpyVector(np.array([1, 2, 3], dtype=np.int16)),
        NumpyVector(np.array([1, 2, 3], dtype=np.int32)),
    )
    # Missing values are ignored
    assert NumpyVector.Type.compare_objects(
        NumpyVector(
            np.array([1, 2, 3, 4]), missing_mask=np.array([False, True, False, True])
        ),
        NumpyVector(
            np.array([1, 2, 3, 22]), missing_mask=np.array([False, True, False, True])
        ),
    )
    # Different missing values
    assert not NumpyVector.Type.compare_objects(
        NumpyVector(
            np.array([1, 2, 3, 4]), missing_mask=np.array([False, False, True, True])
        ),
        NumpyVector(
            np.array([1, 2, 3, 4]), missing_mask=np.array([False, False, False, True])
        ),
    )
    # Coincidental equality of non-missing values is not equality
    assert not NumpyVector.Type.compare_objects(
        NumpyVector(
            np.array([1, 2, 3, 3]), missing_mask=np.array([False, False, True, False])
        ),
        NumpyVector(
            np.array([1, 2, 3, 3]), missing_mask=np.array([False, False, False, True])
        ),
    )
    with pytest.raises(TypeError):
        NumpyVector.Type.compare_objects(5, 5)


def test_graphblas():
    assert GrblasVectorType.compare_objects(
        grblas.Vector.from_values([0, 1, 2], [1, 2, 3]),
        grblas.Vector.from_values([0, 1, 2], [1, 2, 3]),
    )
    assert GrblasVectorType.compare_objects(
        grblas.Vector.from_values([0, 1, 2], [1.1, 2.2, 3.333333333333333333]),
        grblas.Vector.from_values(
            [0, 1, 2], [1.1, 2.19999999999999999, 3.333333333333333334]
        ),
    )
    # Different size
    assert not GrblasVectorType.compare_objects(
        grblas.Vector.from_values([0, 1, 2], [1, 2, 3]),
        grblas.Vector.from_values([0, 1, 2, 3], [1, 2, 3, 4]),
    )
    # Different dtypes are not equal
    assert not GrblasVectorType.compare_objects(
        grblas.Vector.from_values([0, 1, 2], [1, 2, 3], dtype=grblas.dtypes.INT16),
        grblas.Vector.from_values([0, 1, 2], [1, 2, 3], dtype=grblas.dtypes.INT32),
    )
    # Sparse vector
    assert GrblasVectorType.compare_objects(
        grblas.Vector.from_values([0, 2], [1, 3], size=4),
        grblas.Vector.from_values([0, 2], [1, 3], size=4),
    )
    # Different missing values
    assert not GrblasVectorType.compare_objects(
        grblas.Vector.from_values([0, 1, 3], [1, 2, 4], size=4),
        grblas.Vector.from_values([0, 1, 2, 3], [1, 2, 3, 4], size=4),
    )
    # Coincidental equality of non-missing values is not equality
    assert not GrblasVectorType.compare_objects(
        grblas.Vector.from_values([0, 2], [1, 3], size=4),
        grblas.Vector.from_values([0, 3], [1, 3], size=4),
    )
    with pytest.raises(TypeError):
        GrblasVectorType.compare_objects(5, 5)
