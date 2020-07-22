from metagraph import concrete_algorithm, NodeID
from metagraph.plugins import has_scipy
from .types import ScipyEdgeSet, ScipyEdgeMap, ScipyGraph
from typing import Tuple


if has_scipy:
    import scipy.sparse as ss
    from ..numpy.types import NumpyNodeMap, NumpyVector

    @concrete_algorithm("clustering.connected_components")
    def ss_connected_components(graph: ScipyGraph) -> NumpyNodeMap:
        _, node_labels = ss.csgraph.connected_components(
            graph.edges.value, False, return_labels=True
        )
        return NumpyNodeMap(node_labels, node_ids=graph.edges.node_list)

    @concrete_algorithm("clustering.strongly_connected_components")
    def ss_strongly_connected_components(graph: ScipyGraph) -> NumpyNodeMap:
        _, node_labels = ss.csgraph.connected_components(
            graph.edges.value, True, connection="strong", return_labels=True
        )
        return NumpyNodeMap(node_labels, node_ids=graph.edges.node_list)

    @concrete_algorithm("traversal.all_pairs_shortest_paths")
    def ss_all_pairs_shortest_paths(
        graph: ScipyGraph,
    ) -> Tuple[ScipyGraph, ScipyGraph]:
        is_directed = ScipyGraph.Type.compute_abstract_properties(
            graph, {"is_directed"}
        )["is_directed"]
        lengths, parents = ss.csgraph.dijkstra(
            graph.edges.value, directed=is_directed, return_predecessors=True
        )
        lengths = ss.csr_matrix(lengths)
        parents = ss.csr_matrix(parents)
        parents = parents + 9999 * ss.eye(parents.get_shape()[0])
        parents = parents.astype(graph.edges.value.dtype)
        return (
            ScipyGraph(ScipyEdgeMap(parents, graph.edges.node_list), nodes=graph.nodes),
            ScipyGraph(ScipyEdgeMap(lengths, graph.edges.node_list), nodes=graph.nodes),
        )

    @concrete_algorithm("cluster.triangle_count")
    def ss_triangle_count(graph: ScipyGraph) -> int:
        """
        Uses the triangle counting method descripbed in
        https://www.sandia.gov/~srajama/publications/Tricount-HPEC.pdf
        """
        L = ss.tril(graph.edges.value, k=-1).tocsr()
        U = ss.triu(graph.edges.value, k=1).tocsc()
        return int((L @ U.T).multiply(L).sum())

    @concrete_algorithm("traversal.bfs_iter")
    def ss_breadth_first_search_iter(
        graph: ScipyGraph, source_node: NodeID, depth_limit: int
    ) -> NumpyVector:
        is_directed = ScipyGraph.Type.compute_abstract_properties(
            graph, {"is_directed"}
        )["is_directed"]
        bfs_ordered_incides = ss.csgraph.breadth_first_order(
            graph.edges.value,
            source_node,
            directed=is_directed,
            return_predecessors=False,
        )
        bfs_ordered_nodes = graph.edges.node_list[bfs_ordered_incides]
        return NumpyVector(bfs_ordered_nodes)
