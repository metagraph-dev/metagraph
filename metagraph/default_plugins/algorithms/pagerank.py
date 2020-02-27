from ... import abstract_algorithm, concrete_algorithm
from ..abstract_types import Graph, SparseVector
from .. import numpy, networkx, cugraph


@abstract_algorithm("link_analysis.pagerank")
def pagerank(
    graph: Graph, damping: float = 0.85, maxiter: int = 50, tolerance: float = 1e-05
) -> SparseVector:
    pass


if networkx and numpy:
    nx = networkx
    from ..wrappers.python import PythonSparseVector

    @concrete_algorithm("link_analysis.pagerank")
    def nx_pagerank(
        graph: nx.DiGraph,
        damping: float = 0.85,
        maxiter: int = 50,
        tolerance: float = 1e-05,
    ) -> PythonSparseVector:
        pagerank = nx.pagerank(
            graph, alpha=damping, max_iter=maxiter, tol=tolerance, weight=None
        )
        return PythonSparseVector(pagerank)


if cugraph and numpy:
    np = numpy
    from ..wrappers.numpy import NumpySparseVector

    @concrete_algorithm("link_analysis.pagerank")
    def cugraph_pagerank(
        graph: cugraph.DiGraph,
        damping: float = 0.85,
        maxiter: int = 50,
        tolerance: float = 1e-05,
    ) -> NumpySparseVector:
        pagerank = cugraph.pagerank(
            graph, alpha=damping, max_iter=maxiter, tol=tolerance
        )
        out = np.full((graph.number_of_nodes(),), np.nan)
        out[pagerank["vertex"]] = pagerank["pagerank"]
        return NumpySparseVector(out, missing_value=np.nan)
