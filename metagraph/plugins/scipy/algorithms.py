from metagraph import concrete_algorithm
from metagraph.plugins import has_scipy
from .types import ScipyEdgeMap
from typing import Tuple


if has_scipy:
    import scipy.sparse as ss

    @concrete_algorithm("traversal.all_shortest_paths")
    def ss_all_shortest_lengths(
        graph: ScipyEdgeMap,
    ) -> Tuple[ScipyEdgeMap, ScipyEdgeMap]:
        graph_csr = graph.value.tocsr()
        lengths, parents = ss.csgraph.dijkstra(graph_csr, return_predecessors=True)
        lengths = ss.csr_matrix(lengths)
        parents = ss.csr_matrix(parents)
        parents = parents + 9999 * ss.eye(parents.get_shape()[0])
        parents = parents.astype(graph_csr.dtype)
        return (ScipyEdgeMap(parents), ScipyEdgeMap(lengths))

    @concrete_algorithm("cluster.triangle_count")
    def ss_triangle_count(graph: ScipyEdgeMap) -> int:
        """
        Uses the triangle counting method descripbed in
        https://www.sandia.gov/~srajama/publications/Tricount-HPEC.pdf
        """
        L = ss.tril(graph.value, k=-1).tocsr()
        U = ss.triu(graph.value, k=1).tocsc()
        return int((L @ U.T).multiply(L).sum())
