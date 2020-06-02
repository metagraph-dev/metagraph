from metagraph import translator
from metagraph.plugins import has_scipy, has_networkx, has_grblas

if has_scipy:
    import scipy.sparse as ss
    from .types import ScipyEdgeMap, ScipyMatrixType
    from ..numpy.types import NumpyMatrix

    @translator
    def matrix_from_numpy(x: NumpyMatrix, **props) -> ScipyMatrixType:
        # scipy.sparse assumes zero mean empty
        # To work around this limitation, we use a mask
        # and directly set .data after construction
        if x.missing_mask is None:
            mat = ss.coo_matrix(x)
            nrows, ncols = mat.shape
            if mat.nnz != nrows * ncols:
                mat.data = x.value.flatten()
        else:
            mat = ss.coo_matrix(~x.missing_mask)
            mat.data = x.value[~x.missing_mask]
        return mat


if has_scipy and has_networkx:
    import networkx as nx
    from .types import ScipyMatrixType, ScipyEdgeMap
    from ..networkx.types import NetworkXEdgeMap

    @translator
    def edgemap_from_networkx(x: NetworkXEdgeMap, **props) -> ScipyEdgeMap:
        ordered_nodes = list(sorted(x.value.nodes()))
        m = nx.convert_matrix.to_scipy_sparse_matrix(x.value, nodelist=ordered_nodes)
        return ScipyEdgeMap(m, ordered_nodes)


if has_scipy and has_grblas:
    import scipy.sparse as ss
    from ..graphblas.types import GrblasMatrixType

    @translator
    def matrix_from_graphblas(x: GrblasMatrixType, **props) -> ScipyMatrixType:
        rows, cols, vals = x.to_values()
        mat = ss.coo_matrix((vals, (rows, cols)), x.shape)
        return mat
