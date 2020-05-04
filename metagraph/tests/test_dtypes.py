import pytest
from metagraph.core import dtypes
import numpy as np


def test_dtype():
    with pytest.raises(ValueError):
        dtypes.dtype("U4")

    assert dtypes.dtype("bool") == dtypes.bool
    assert dtypes.dtype(np.int32) == dtypes.int32
