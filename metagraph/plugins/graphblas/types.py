from metagraph import ConcreteType, Wrapper, dtypes
from metagraph.types import SparseVector, SparseMatrix, Graph, WeightedGraph
from metagraph.plugins import has_grblas


if has_grblas:
    import grblas

    dtype_grblas_to_mg = {
        grblas.dtypes.BOOL: dtypes.bool,
        grblas.dtypes.INT8: dtypes.int8,
        grblas.dtypes.INT16: dtypes.int16,
        grblas.dtypes.INT32: dtypes.int32,
        grblas.dtypes.INT64: dtypes.int64,
        grblas.dtypes.UINT8: dtypes.uint8,
        grblas.dtypes.UINT16: dtypes.uint16,
        grblas.dtypes.UINT32: dtypes.uint32,
        grblas.dtypes.UINT64: dtypes.uint64,
        grblas.dtypes.FP32: dtypes.float32,
        grblas.dtypes.FP64: dtypes.float64,
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

        def show(self):
            return self.value.show()

    class GrblasWeightedAdjacencyMatrix(Wrapper, abstract=WeightedGraph):
        def __init__(self, value, transposed=False):
            self.value = value
            self.transposed = transposed
            self._assert_instance(value, grblas.Matrix)

        def shape(self):
            return (self.value.nrows, self.value.ncols)

        def dtype(self):
            return dtype_grblas_to_mg[self.value.dtype]

        def show(self):
            return self.value.show()

    class GrblasIncidenceMatrix(Wrapper, abstract=Graph):
        def __init__(self, value, transposed=False):
            self.value = value
            self.transposed = transposed
            self._assert_instance(value, grblas.Matrix)

        def shape(self):
            return (self.value.nrows, self.value.ncols)

        def dtype(self):
            return dtype_grblas_to_mg[self.value.dtype]

        def show(self):
            return self.value.show()
