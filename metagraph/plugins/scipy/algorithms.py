import numpy as np
from metagraph import concrete_algorithm, NodeID
from metagraph.plugins import has_scipy
from .types import ScipyEdgeSet, ScipyEdgeMap, ScipyGraph
from .. import has_numba
import numpy as np
from typing import Tuple, Callable, Any, Union

if has_numba:
    import numba

if has_scipy:
    import scipy.sparse as ss
    from ..numpy.types import NumpyNodeMap, NumpyNodeSet, NumpyVector

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

    @concrete_algorithm("traversal.minimum_spanning_tree")
    def ss_minimum_spanning_tree(graph: ScipyGraph) -> ScipyGraph:
        span_tree = ss.csgraph.minimum_spanning_tree(graph.edges.value)
        span_tree_mask = (span_tree != 0).astype(int, copy=False)
        span_tree_mask_transposed = span_tree_mask.T
        divisor = span_tree_mask + span_tree_mask_transposed
        undirected_span_tree = span_tree + span_tree.T
        # assert (undirected_span_tree.indices == divisor.indices).all()
        # assert (undirected_span_tree.indptr == divisor.indptr).all()
        undirected_span_tree.data /= divisor.data
        undirected_span_tree = undirected_span_tree.astype(
            graph.edges.value.dtype, copy=False
        )
        return ScipyGraph(undirected_span_tree, nodes=graph.nodes)

    @concrete_algorithm("cluster.triangle_count")
    def ss_triangle_count(graph: ScipyGraph) -> int:
        """
        Uses the triangle counting method described in
        https://www.sandia.gov/~srajama/publications/Tricount-HPEC.pdf
        """
        props = ScipyGraph.Type.compute_abstract_properties(graph, {"edge_type"})
        if props["edge_type"] == "map":
            # Drop weights before performing triangle count
            m = graph.edges.value.copy()
            m.data = np.ones_like(m.data)
        elif props["edge_type"] == "set":
            m = graph.edges.value
        L = ss.tril(m, k=-1).tocsr()
        U = ss.triu(m, k=1).tocsc()
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

    @concrete_algorithm("flow.max_flow")
    def ss_max_flow(
        graph: ScipyGraph, source_node: NodeID, target_node: NodeID,
    ) -> Tuple[float, ScipyGraph]:
        max_flow_result = ss.csgraph.maximum_flow(
            graph.edges.value, source_node, target_node
        )
        flow_value = max_flow_result.flow_value
        residual_graph = max_flow_result.residual
        residual_keep_mask = residual_graph > 0
        ss_flow_graph = residual_graph.multiply(residual_keep_mask)
        flow_graph = ScipyGraph(ss_flow_graph, nodes=graph.nodes)
        return (flow_value, flow_graph)

    def _reduce_sparse_matrix(
        func: np.ufunc, sparse_matrix: ss.spmatrix
    ) -> Tuple[np.ndarray, np.ndarray]:
        keep_mask = np.diff(sparse_matrix.indptr).astype(bool)
        reduceat_indices = sparse_matrix.indptr[:-1][keep_mask]
        reduced_values = func.reduceat(
            sparse_matrix.data, reduceat_indices, dtype=object
        )
        return reduced_values, keep_mask

    @concrete_algorithm("util.graph.aggregate_edges")
    def ss_graph_aggregate_edges(
        graph: ScipyGraph,
        func: Callable[[Any, Any], Any],
        initial_value: Any,
        in_edges: bool,
        out_edges: bool,
    ) -> NumpyNodeMap:
        if in_edges or out_edges:
            is_directed = ScipyGraph.Type.compute_abstract_properties(
                graph, {"is_directed"}
            )["is_directed"]
            if not is_directed:
                in_edges = True
                out_edges = False
        nrows = graph.edges.value.shape[0]
        num_agg_values = nrows if graph.nodes is None else len(graph.nodes)
        final_position_to_agg_value = np.full(num_agg_values, initial_value)
        if not isinstance(func, np.ufunc):
            func = np.frompyfunc(func, 2, 1)
        matrix_position_to_agg_value = np.full(nrows, initial_value)
        if in_edges:
            csc_matrix = graph.edges.value.tocsc()
            in_edges_aggregated_values, keep_mask = _reduce_sparse_matrix(
                func, csc_matrix
            )
            matrix_position_to_agg_value[keep_mask] = func(
                matrix_position_to_agg_value[keep_mask], in_edges_aggregated_values
            )
        if out_edges:
            csr_matrix = graph.edges.value
            out_edges_aggregated_values, keep_mask = _reduce_sparse_matrix(
                func, csr_matrix
            )
            matrix_position_to_agg_value[keep_mask] = func(
                matrix_position_to_agg_value[keep_mask], out_edges_aggregated_values
            )
        # TODO This doesn't assume sortedness of any node list ; make these other data structures not require sorted node lists as that is expensive for large graphs
        graph_node_ids = (
            graph.edges.node_list if graph.nodes is None else graph.nodes.nodes()
        )
        matrix_position_to_node_id = graph.edges.node_list
        graph_node_ids_position_to_final_position = np.argsort(graph_node_ids)
        final_position_to_graph_node_id = graph_node_ids[
            graph_node_ids_position_to_final_position
        ]
        matrix_position_to_final_position = np.searchsorted(
            final_position_to_graph_node_id, matrix_position_to_node_id
        )
        final_position_to_agg_value[
            matrix_position_to_final_position
        ] = matrix_position_to_agg_value
        # Would we ever want to return a NumpyNodeMap via a mask?
        return NumpyNodeMap(
            final_position_to_agg_value, node_ids=final_position_to_graph_node_id
        )

    @concrete_algorithm("util.graph.filter_edges")
    def ss_graph_filter_edges(
        graph: ScipyGraph, func: Callable[[Any], bool]
    ) -> ScipyGraph:
        # TODO consider caching this somewhere or enforcing that only vectorized functions are given
        func_vectorized = numba.vectorize(func) if has_numba else np.vectorize(func)
        # TODO Explicitly handle the CSR case
        result_matrix = (
            graph.edges.value.copy()
            if isinstance(graph.edges.value, ss.coo_matrix)
            else graph.edges.value.tocoo(copy=True)
        )
        result_edge_map = ScipyEdgeMap(
            result_matrix, graph.edges.node_list, graph.edges.transposed
        )
        to_keep_mask = func_vectorized(result_edge_map.value.data)
        if not to_keep_mask.all():
            result_edge_map.value.row = result_edge_map.value.row[to_keep_mask]
            result_edge_map.value.col = result_edge_map.value.col[to_keep_mask]
            result_edge_map.value.data = result_edge_map.value.data[to_keep_mask]
        result_graph_nodes = None if graph.nodes is None else graph.nodes.copy()
        return ScipyGraph(result_edge_map, result_graph_nodes)

    @concrete_algorithm("util.graph.assign_uniform_weight")
    def ss_graph_assign_uniform_weight(graph: ScipyGraph, weight: Any) -> ScipyGraph:
        matrix = graph.edges.value.copy()
        matrix.data.fill(weight)
        edge_map = ScipyEdgeMap(
            matrix, node_list=graph.edges.node_list, transposed=graph.edges.transposed
        )
        nodes = None if graph.nodes is None else graph.nodes.copy()
        return ScipyGraph(edge_map, nodes=nodes)

    @concrete_algorithm("util.graph.build")
    def ss_graph_build(
        edges: Union[ScipyEdgeSet, ScipyEdgeMap],
        nodes: Union[NumpyNodeSet, NumpyNodeMap, None],
    ) -> ScipyGraph:
        return ScipyGraph(edges, nodes)

    @concrete_algorithm("util.edgemap.from_edgeset")
    def ss_edgemap_from_edgeset(
        edgeset: ScipyEdgeSet, default_value: Any,
    ) -> ScipyEdgeMap:
        new_matrix = edgeset.value.copy()
        new_matrix.data.fill(default_value)
        return ScipyEdgeMap(new_matrix, edgeset.node_list.copy(), edgeset.transposed)
