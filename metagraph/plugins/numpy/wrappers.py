import numpy as np
from metagraph import Wrapper, dtypes
from metagraph.types import DenseVector, SparseVector, DenseMatrix, SparseMatrix


def dtype_mg_to_np(dtype):
    return dtypes.dtype(dtype)


dtype_np_to_mg = dtype_mg_to_np


class NumpyVector(Wrapper, abstract=DenseVector):
    def __init__(self, data):
        self.value = data
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")

    def __len__(self):
        return len(self.value)

    @property
    def dtype(self):
        return dtype_np_to_mg(self.value.dtype)


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


class NumpyMatrix(Wrapper, abstract=DenseMatrix):
    def __init__(self, data):
        """
        data: np.ndarray with ndims=2
              if passed np.matrix, will convert to np.ndarray
        """
        if isinstance(data, np.matrix):
            data = np.array(data)
        self.value = data
        self._assert_instance(data, np.ndarray)

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return dtype_np_to_mg(self.value.dtype)


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
