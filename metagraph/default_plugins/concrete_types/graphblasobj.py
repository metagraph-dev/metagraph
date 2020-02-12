from ... import PluginRegistry, ConcreteType, dtypes
from ..abstract_types import (
    SparseVectorType,
    SparseMatrixType,
    GraphType,
    WeightedGraphType,
)

reg = PluginRegistry("metagraph_core")

try:
    import grblas

    grblas.init("suitesparse")
except ImportError:
    grblas = None


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

    # class GrblasVector(SparseArray):
    #     def __init__(self, obj):
    #         self.obj = obj
    #         assert isinstance(obj, grblas.Vector)
    #
    #     def __len__(self):
    #         return self.obj.size
    #
    #     def dtype(self):
    #         return dtype_grblas_to_mg[self.obj.dtype]

    class GrblasAdjacencyMatrix:
        def __init__(self, obj, transposed=False):
            self.obj = obj
            self.transposed = transposed
            assert isinstance(obj, grblas.Matrix)

        def shape(self):
            return (self.obj.nrows, self.obj.ncols)

        def dtype(self):
            return dtype_grblas_to_mg[self.obj.dtype]

    class GrblasWeightedAdjacencyMatrix:
        def __init__(self, obj, transposed=False):
            self.obj = obj
            self.transposed = transposed
            assert isinstance(obj, grblas.Matrix)

        def shape(self):
            return (self.obj.nrows, self.obj.ncols)

        def dtype(self):
            return dtype_grblas_to_mg[self.obj.dtype]

    class GrblasIncidenceMatrix:
        def __init__(self, obj, transposed=False):
            self.obj = obj
            self.transposed = transposed
            assert isinstance(obj, grblas.Matrix)

        def shape(self):
            return (self.obj.nrows, self.obj.ncols)

        def dtype(self):
            return dtype_grblas_to_mg[self.obj.dtype]

    @reg.register
    class GrblasVectorType(ConcreteType):
        name = "GrblasVector"
        abstract = SparseVectorType
        value_class = grblas.Vector

    @reg.register
    class GrblasMatrixType(ConcreteType):
        name = "GrblasMatrix"
        abstract = SparseMatrixType
        value_class = grblas.Matrix

    @reg.register
    class GrblasAdjacencyMatrixType(ConcreteType):
        name = "GrblasAdjacencyMatrix"
        abstract = GraphType
        value_class = GrblasAdjacencyMatrix

    @reg.register
    class GrblasWeightedAdjacencyMatrixType(ConcreteType):
        name = "GrblasWeightedAdjacencyMatrix"
        abstract = WeightedGraphType
        value_class = GrblasWeightedAdjacencyMatrix

    @reg.register
    class GrblasIncidenceMatrix(ConcreteType):
        name = "GrblasIncidenceMatrix"
        abstract = GraphType
        value_class = GrblasIncidenceMatrix
