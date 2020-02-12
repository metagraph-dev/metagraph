from ... import PluginRegistry, ConcreteType
from ..abstract_types import SparseMatrixType, GraphType, WeightedGraphType

reg = PluginRegistry("metagraph_core")

try:
    import scipy.sparse as ss
except ImportError:
    ss = None


if ss is not None:

    class ScipySparseMatrix:
        def __init__(self, data):
            self.obj = data
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.obj.format

    class ScipyAdjacencyMatrix:
        def __init__(self, data, transposed=False):
            self.obj = data
            self.transposed = transposed
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.obj.format

    class ScipyWeightedAdjacencyMatrix:
        def __init__(self, data, transposed=False):
            self.obj = data
            self.transposed = transposed
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.obj.format

    class ScipyIncidenceMatrix:
        def __init__(self, data, transposed=False):
            self.obj = data
            self.transposed = transposed
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.obj.format

    @reg.register
    class ScipySparseMatrixType(ConcreteType):
        name = "ScipySparseMatrix"
        abstract = SparseMatrixType
        value_class = ScipySparseMatrix

    @reg.register
    class ScipyAdjacencyMatrix(ConcreteType):
        name = "ScipyAdjacencyMatrix"
        abstract = GraphType
        value_class = ScipyAdjacencyMatrix

    @reg.register
    class ScipyWeightedAdjacencyMatrixType(ConcreteType):
        name = "ScipyWeightedAdjacencyMatrix"
        abstract = WeightedGraphType
        value_class = ScipyWeightedAdjacencyMatrix

    @reg.register
    class ScipyIncidenceMatrixType(ConcreteType):
        name = "ScipyIncidenceMatrix"
        abstract = GraphType
        value_class = ScipyIncidenceMatrix
