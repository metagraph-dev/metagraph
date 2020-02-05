from .base import DenseArray, DenseMatrix
from .. import dtypes

try:
    import numpy as np
except ImportError:
    np = None


if np is not None:
    dtype_np_to_mg = {
        np.bool_: dtypes.BOOL,
        np.int8: dtypes.INT8,
        np.int16: dtypes.INT16,
        np.int32: dtypes.INT32,
        np.int64: dtypes.INT64,
        np.uint8: dtypes.UINT8,
        np.uint16: dtypes.UINT16,
        np.uint32: dtypes.UINT32,
        np.uint64: dtypes.UINT64,
        np.float32: dtypes.FLOAT32,
        np.float64: dtypes.FLOAT64,
    }
    dtype_mg_to_np = {v: k for k, v in dtype_np_to_mg.items()}

    class NumpyDenseArray(DenseArray):
        def __init__(self, data, missing_value=np.nan):
            super().__init__(missing_value)
            self.obj = data
            assert isinstance(data, np.ndarray)
            assert len(data.shape) == 1

        def __len__(self):
            return len(self.obj)

        @property
        def dtype(self):
            return dtype_np_to_mg[self.obj.dtype.type]

    class NumpyDenseMatrix(DenseMatrix):
        def __init__(self, data, missing_value=np.nan):
            """
            data: np.matrix
            """
            super().__init__(missing_value)
            self.obj = data
            self.missing_value = missing_value
            assert isinstance(data, np.matrix)

        @property
        def shape(self):
            return self.obj.shape

        @property
        def dtype(self):
            return dtype_np_to_mg[self.obj.dtype.type]
