from ... import ConcreteType, Wrapper
from ..abstract_types import SparseMatrix, Graph, WeightedGraph
from .. import scipy


if scipy is not None:
    ss = scipy.sparse

    class ScipySparseMatrixType(ConcreteType, abstract=SparseMatrix):
        value_type = ss.spmatrix

    class ScipyAdjacencyMatrix(Wrapper, abstract=Graph):
        def __init__(self, data, transposed=False):
            self.value = data
            self.transposed = transposed
            self._assert_instance(data, ss.spmatrix)

        @property
        def format(self):
            return self.value.format

    class ScipyWeightedAdjacencyMatrix(Wrapper, abstract=WeightedGraph):
        def __init__(self, data, transposed=False):
            self.value = data
            self.transposed = transposed
            self._assert_instance(data, ss.spmatrix)

        @property
        def format(self):
            return self.value.format

    class ScipyIncidenceMatrix(Wrapper, abstract=Graph):
        def __init__(self, data, transposed=False):
            self.value = data
            self.transposed = transposed
            self._assert_instance(data, ss.spmatrix)

        @property
        def format(self):
            return self.value.format
