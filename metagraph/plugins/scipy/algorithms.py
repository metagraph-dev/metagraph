from metagraph import concrete_algorithm, NodeID
from metagraph.plugins import has_scipy
from .types import ScipyEdgeSet, ScipyEdgeMap, ScipyGraph
from .. import has_numba
import numpy as np
from typing import Tuple, Callable, Any

if has_numba:
    import numba

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

    @concrete_algorithm("util.graph.filter_edges")
    def ss_graph_filter_edges(
        graph: ScipyGraph, func: Callable[[Any], bool]
    ) -> ScipyGraph:
        # TODO consider caching this somewhere or enforcing that only vectorized functions are given
        func_vectorized = numba.vectorize(func) if has_numba else np.vectorize(func)
        result_edge_map = graph.edges.copy()
        to_remove_mask = ~func_vectorized(result_edge_map.value.data)
        if to_remove_mask.any():
            result_edge_map.value.data[to_remove_mask] = 0
            result_edge_map.value.eliminate_zeros()
        result_graph_nodes = graph.nodes if graph.nodes is None else graph.nodes.copy()
        return ScipyGraph(result_edge_map, result_graph_nodes)

    @concrete_algorithm("util.graph.add_uniform_weight")
    def ss_graph_add_uniform_weight(graph: ScipyGraph, weight: Any) -> ScipyGraph:
        result = graph.copy()
        nonzero_row_col_tuple = result.edges.value.nonzero()
        num_nonzero_elems = len(nonzero_row_col_tuple[0])
        result.edges.value = result.edges.value + ss.csr_matrix(
            (np.full(num_nonzero_elems, weight), nonzero_row_col_tuple),
            result.edges.value.shape,
        )
        return result
