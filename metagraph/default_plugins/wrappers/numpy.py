from ... import Wrapper, dtypes
from ..abstract_types import DenseVector, SparseVector, DenseMatrix, SparseMatrix
from .. import registry, numpy


if numpy is not None:
    np = numpy

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

    @registry.register
    class NumpyVector(Wrapper, abstract=DenseVector):
        def __init__(self, data):
            self.value = data
            assert isinstance(data, np.ndarray)
            assert len(data.shape) == 1

        def __len__(self):
            return len(self.value)

        @property
        def dtype(self):
            return dtype_np_to_mg[self.value.dtype.type]

    @registry.register
    class NumpySparseVector(NumpyVector, abstract=SparseVector):
        def __init__(self, data, missing_value=np.nan):
            super().__init__(data)
            self.missing_value = missing_value

        def get_missing_mask(self):
            """
            Returns an array of True/False where True indicates a missing value
            """
            if self.missing_value != self.missing_value:
                # Special handling for np.nan which does not equal itself
                return np.isnan(self.value)
            else:
                return self.value == self.missing_value

        @property
        def nnz(self):
            return np.count_nonzero(~self.get_missing_mask())

    @registry.register
    class NumpyMatrix(Wrapper, abstract=DenseMatrix):
        def __init__(self, data):
            """
            data: np.ndarray with ndims=2
                  if passed np.matrix, will convert to np.ndarray
            """
            if isinstance(data, np.matrix):
                data = np.array(data)
            self.value = data
            assert isinstance(data, np.ndarray)

        @property
        def shape(self):
            return self.value.shape

        @property
        def dtype(self):
            return dtype_np_to_mg[self.value.dtype.type]

    @registry.register
    class NumpySparseMatrix(NumpyMatrix, abstract=SparseMatrix):
        def __init__(self, data, missing_value=np.nan):
            super().__init__(data)
            self.missing_value = missing_value

        def get_missing_mask(self):
            """
            Returns an array of True/False where True indicates a missing value
            """
            if self.missing_value != self.missing_value:
                # Special handling for np.nan which does not equal itself
                return np.isnan(self.value)
            else:
                return self.value == self.missing_value

        @property
        def nnz(self):
            return np.count_nonzero(~self.get_missing_mask())
