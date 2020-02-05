from .base import (
    SparseMatrix,
    AdjacencyMatrix,
    WeightedAdjacencyMatrix,
    IncidenceMatrix,
)

try:
    import scipy.sparse as ss
except ImportError:
    ss = None


if ss is not None:

    class ScipySparseMatrix(SparseMatrix):
        def __init__(self, data):
            self.obj = data
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.obj.format

    class ScipyAdjacencyMatrix(AdjacencyMatrix):
        def __init__(self, data, transposed=False):
            super().__init__(transposed)
            self.obj = data
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.obj.format

    class ScipyWeightedAdjacencyMatrix(WeightedAdjacencyMatrix):
        def __init__(self, data, transposed=False):
            super().__init__(transposed)
            self.obj = data
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.obj.format

    class ScipyIncidenceMatrix(IncidenceMatrix):
        def __init__(self, data, transposed=False):
            super().__init__(transposed)
            self.obj = data
            assert isinstance(data, ss.spmatrix)

        @property
        def format(self):
            return self.obj.format
