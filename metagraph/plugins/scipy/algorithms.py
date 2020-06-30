from metagraph import concrete_algorithm, NodeID
from metagraph.plugins import has_scipy
from .types import ScipyEdgeSet, ScipyEdgeMap
from typing import Tuple


if has_scipy:
    import scipy.sparse as ss
    from ..numpy.types import NumpyNodeMap, NumpyVector

    @concrete_algorithm("clustering.connected_components")
    def ss_connected_components(graph: ScipyEdgeSet) -> NumpyNodeMap:
        _, node_labels = ss.csgraph.connected_components(
            graph.value, False, return_labels=True
        )
        return graph.node_id_map_from_index_map(node_labels)

    @concrete_algorithm("clustering.strongly_connected_components")
    def ss_strongly_connected_components(graph: ScipyEdgeSet) -> NumpyNodeMap:
        _, node_labels = ss.csgraph.connected_components(
            graph.value, True, connection="strong", return_labels=True
        )
        return graph.node_id_map_from_index_map(node_labels)

    @concrete_algorithm("traversal.all_shortest_paths")
    def ss_all_shortest_lengths(
        graph: ScipyEdgeMap,
    ) -> Tuple[ScipyEdgeMap, ScipyEdgeMap]:
        is_directed = ScipyEdgeMap.Type.compute_abstract_properties(
            graph, {"is_directed"}
        )["is_directed"]
        lengths, parents = ss.csgraph.dijkstra(
            graph.value, directed=is_directed, return_predecessors=True
        )
        lengths = ss.csr_matrix(lengths)
        parents = ss.csr_matrix(parents)
        parents = parents + 9999 * ss.eye(parents.get_shape()[0])
        parents = parents.astype(graph.value.dtype)
        return (
            ScipyEdgeMap(parents, graph.node_list),
            ScipyEdgeMap(lengths, graph.node_list),
        )

    @concrete_algorithm("cluster.triangle_count")
    def ss_triangle_count(graph: ScipyEdgeSet) -> int:
        """
        Uses the triangle counting method descripbed in
        https://www.sandia.gov/~srajama/publications/Tricount-HPEC.pdf
        """
        L = ss.tril(graph.value, k=-1).tocsr()
        U = ss.triu(graph.value, k=1).tocsc()
        return int((L @ U.T).multiply(L).sum())

    @concrete_algorithm("traversal.breadth_first_search")
    def ss_breadth_first_search(
        graph: ScipyEdgeSet, source_node: NodeID
    ) -> NumpyVector:
        is_directed = ScipyEdgeMap.Type.compute_abstract_properties(
            graph, {"is_directed"}
        )["is_directed"]
        bfs_ordered_incides = ss.csgraph.breadth_first_order(
            graph.value, source_node, directed=is_directed, return_predecessors=False
        )
        bfs_ordered_nodes = graph.node_list[bfs_ordered_incides]
        return NumpyVector(bfs_ordered_nodes)
