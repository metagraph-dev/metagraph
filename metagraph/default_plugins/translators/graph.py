from ... import translator
from .. import scipy, pandas, networkx, grblas


if pandas and networkx:
    pd = pandas
    nx = networkx
    from ..wrappers.pandas import PandasEdgeList
    from ..wrappers.networkx import NetworkXGraphType

    @translator
    def translate_graph_pdedge2nx(x: PandasEdgeList, **props) -> NetworkXGraphType:
        g = x.value[[x.src_label, x.dest_label]]
        out = nx.DiGraph()
        out.add_edges_from(g.itertuples(index=False, name="Edge"))
        return out

    @translator
    def translate_graph_nx2pdedge(x: NetworkXGraphType, **props) -> PandasEdgeList:
        df = nx.convert_matrix.to_pandas_edgelist(
            x, source="source", target="destination"
        )
        return PandasEdgeList(df, src_label="source", dest_label="destination")


if networkx and scipy:
    nx = networkx
    ss = scipy.sparse
    from ..wrappers.networkx import NetworkXGraphType
    from ..wrappers.scipy import ScipyAdjacencyMatrix

    @translator
    def translate_graph_nx2scipy(x: NetworkXGraphType, **props) -> ScipyAdjacencyMatrix:
        # WARNING: This assumes the nxGraph has nodes in sequential order
        m = nx.convert_matrix.to_scipy_sparse_matrix(x, nodelist=range(len(x)))
        return ScipyAdjacencyMatrix(m)


if scipy and grblas:
    ss = scipy.sparse
    from ..wrappers.scipy import ScipyAdjacencyMatrix
    from ..wrappers.graphblas import GrblasAdjacencyMatrix

    @translator
    def translate_graph_pdedge2grblas(
        x: ScipyAdjacencyMatrix, **props
    ) -> GrblasAdjacencyMatrix:
        m = x.value.tocoo()
        nrows, ncols = m.shape
        out = grblas.Matrix.new_from_values(
            m.row, m.col, m.data, nrows=nrows, ncols=ncols, dtype=grblas.dtypes.INT64
        )
        return GrblasAdjacencyMatrix(out, transposed=x.transposed)
