from metagraph import translator
from metagraph.plugins import has_scipy, has_networkx, has_grblas

if has_scipy:
    import scipy.sparse as ss
    from .wrappers import ScipyAdjacencyMatrix, ScipySparseMatrixType
    from ..numpy.wrappers import NumpySparseMatrix

    @translator
    def sparsematrix_from_numpy(x: NumpySparseMatrix, **props) -> ScipySparseMatrixType:
        # scipy.sparse assumes zero mean empty
        # To work around this limitation, we use a mask
        # and directly set .data after construction
        non_mask = ~x.get_missing_mask()
        mat = ss.coo_matrix(non_mask)
        mat.data = x.value[non_mask]
        return mat


if has_scipy and has_networkx:
    import networkx as nx
    from ..networkx.wrappers import NetworkXGraphType

    @translator
    def graph_from_networkx(x: NetworkXGraphType, **props) -> ScipyAdjacencyMatrix:
        # WARNING: This assumes the nxGraph has nodes in sequential order
        m = nx.convert_matrix.to_scipy_sparse_matrix(x, nodelist=range(len(x)))
        return ScipyAdjacencyMatrix(m)


if has_scipy and has_grblas:
    from ..graphblas.wrappers import GrblasMatrixType

    @translator
    def sparsematrix_from_graphblas(
        x: GrblasMatrixType, **props
    ) -> ScipySparseMatrixType:
        rows, cols, vals = x.to_values()
        mat = ss.coo_matrix((tuple(vals), (tuple(rows), tuple(cols))), x.shape)
        return mat
