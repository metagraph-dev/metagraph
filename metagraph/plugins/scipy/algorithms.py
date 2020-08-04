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
    from ..python.types import PythonNodeSet
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
        print()
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
        # TODO This doesn't assume sortedness of any node list ; make these other data structures not require sored node lists as that is expensive for large graphs
        graph_node_ids = graph.edges.node_list if graph.nodes is None else graph.nodes
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

    @concrete_algorithm("util.graph.build")
    def ss_graph_build(
        edges: Union[ScipyEdgeSet, ScipyEdgeMap],
        nodes: Union[PythonNodeSet, NumpyNodeMap, None],
    ) -> ScipyGraph:
        return ScipyGraph(edges, nodes)

    @concrete_algorithm("util.edge_map.from_edge_set")
    def ss_edge_map_from_edge_set(
        edge_set: ScipyEdgeSet, default_value: Any,
    ) -> ScipyEdgeMap:
        new_matrix = edge_set.value.copy()
        new_matrix.data.fill(default_value)
        return ScipyEdgeMap(new_matrix, edge_set.node_list.copy(), edge_set.transposed)
