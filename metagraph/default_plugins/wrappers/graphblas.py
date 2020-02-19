from ... import ConcreteType, Wrapper, dtypes
from ..abstract_types import (
    SparseVector,
    SparseMatrix,
    Graph,
    WeightedGraph,
)
from .. import grblas


if grblas is not None:
    dtype_grblas_to_mg = {
        grblas.dtypes.BOOL: dtypes.BOOL,
        grblas.dtypes.INT8: dtypes.INT8,
        grblas.dtypes.INT16: dtypes.INT16,
        grblas.dtypes.INT32: dtypes.INT32,
        grblas.dtypes.INT64: dtypes.INT64,
        grblas.dtypes.UINT8: dtypes.UINT8,
        grblas.dtypes.UINT16: dtypes.UINT16,
        grblas.dtypes.UINT32: dtypes.UINT32,
        grblas.dtypes.UINT64: dtypes.UINT64,
        grblas.dtypes.FP32: dtypes.FLOAT32,
        grblas.dtypes.FP64: dtypes.FLOAT64,
    }
    dtype_mg_to_grblas = {v: k for k, v in dtype_grblas_to_mg.items()}

    class GrblasVectorType(ConcreteType, abstract=SparseVector):
        value_type = grblas.Vector

    class GrblasMatrixType(ConcreteType, abstract=SparseMatrix):
        value_type = grblas.Matrix

    class GrblasAdjacencyMatrix(Wrapper, abstract=Graph):
        def __init__(self, value, transposed=False):
            self.value = value
            self.transposed = transposed
            self._assert_instance(value, grblas.Matrix)

        def shape(self):
            return (self.value.nrows, self.value.ncols)

        def dtype(self):
            return dtype_grblas_to_mg[self.value.dtype]

    class GrblasWeightedAdjacencyMatrix(Wrapper, abstract=WeightedGraph):
        def __init__(self, value, transposed=False):
            self.value = value
            self.transposed = transposed
            self._assert_instance(value, grblas.Matrix)

        def shape(self):
            return (self.value.nrows, self.value.ncols)

        def dtype(self):
            return dtype_grblas_to_mg[self.value.dtype]

    class GrblasIncidenceMatrix(Wrapper, abstract=Graph):
        def __init__(self, value, transposed=False):
            self.value = value
            self.transposed = transposed
            self._assert_instance(value, grblas.Matrix)

        def shape(self):
            return (self.value.nrows, self.value.ncols)

        def dtype(self):
            return dtype_grblas_to_mg[self.value.dtype]
