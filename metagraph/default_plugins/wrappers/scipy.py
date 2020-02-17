from ... import ConcreteType, Wrapper
from ..abstract_types import SparseMatrix, Graph, WeightedGraph
from .. import registry


try:
    import scipy.sparse as ss
except ImportError:
    ss = None


if ss is not None:

    @registry.register
    class ScipySparseMatrixType(ConcreteType, abstract=SparseMatrix):
        value_type = ss.spmatrix

    @registry.register
    class ScipyAdjacencyMatrix(Wrapper, abstract=Graph):
        def __init__(self, data, transposed=False):
            self.value = data
            self.transposed = transposed
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.value.format

    @registry.register
    class ScipyWeightedAdjacencyMatrix(Wrapper, abstract=WeightedGraph):
        def __init__(self, data, transposed=False):
            self.value = data
            self.transposed = transposed
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.value.format

    @registry.register
    class ScipyIncidenceMatrix(Wrapper, abstract=Graph):
        def __init__(self, data, transposed=False):
            self.value = data
            self.transposed = transposed
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.value.format
