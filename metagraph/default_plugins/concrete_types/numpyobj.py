from ... import PluginRegistry, ConcreteType, dtypes
from ..abstract_types import VectorType, SparseVectorType, MatrixType, SparseMatrixType

reg = PluginRegistry("metagraph_core")

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

    class NumpyVector:
        def __init__(self, data):
            self.obj = data
            assert isinstance(data, np.ndarray)
            assert len(data.shape) == 1

        def __len__(self):
            return len(self.obj)

        @property
        def dtype(self):
            return dtype_np_to_mg[self.obj.dtype.type]

    class NumpySparseVector(NumpyVector):
        def __init__(self, data, missing_value=np.nan):
            super().__init__(data)
            self.missing_value = missing_value

    class NumpyMatrix:
        def __init__(self, data):
            """
            data: np.ndarray with ndims=2
                  if passed np.matrix, will convert to np.ndarray
            """
            if isinstance(data, np.matrix):
                data = np.array(data)
            self.obj = data
            assert isinstance(data, np.ndarray)

        @property
        def shape(self):
            return self.obj.shape

        @property
        def dtype(self):
            return dtype_np_to_mg[self.obj.dtype.type]

    class NumpySparseMatrix(NumpyMatrix):
        def __init__(self, data, missing_value=np.nan):
            super().__init__(data)
            self.missing_value = missing_value

    @reg.register
    class NumpyVectorType(ConcreteType):
        name = "NumpyVector"
        abstract = VectorType
        value_class = NumpyVector

    @reg.register
    class NumpySparseVectorType(ConcreteType):
        name = "NumpySparseVector"
        abstract = SparseVectorType
        value_class = NumpySparseVector

    @reg.register
    class NumpyMatrixType(ConcreteType):
        name = "NumpyMatrix"
        abstract = MatrixType
        value_class = NumpyMatrix

    @reg.register
    class NumpySparseMatrixType(ConcreteType):
        name = "NumpySparseMatrix"
        abstract = SparseMatrixType
        value_class = NumpySparseMatrix
