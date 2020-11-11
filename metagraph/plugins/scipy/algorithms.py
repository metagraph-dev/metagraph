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
    from ..numpy.types import NumpyNodeMap, NumpyNodeSet, NumpyVectorType

    @concrete_algorithm("clustering.connected_components")
    def ss_connected_components(graph: ScipyGraph) -> NumpyNodeMap:
        _, node_labels = ss.csgraph.connected_components(
            graph.value, False, return_labels=True
        )
        return NumpyNodeMap(node_labels, nodes=graph.node_list)

    @concrete_algorithm("clustering.strongly_connected_components")
    def ss_strongly_connected_components(graph: ScipyGraph) -> NumpyNodeMap:
        _, node_labels = ss.csgraph.connected_components(
            graph.value, True, connection="strong", return_labels=True
        )
        return NumpyNodeMap(node_labels, nodes=graph.node_list)

    @concrete_algorithm("traversal.all_pairs_shortest_paths")
    def ss_all_pairs_shortest_paths(
        graph: ScipyGraph,
    ) -> Tuple[ScipyGraph, ScipyGraph]:
        is_directed = ScipyGraph.Type.compute_abstract_properties(
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
            ScipyGraph(parents, graph.node_list),
            ScipyGraph(lengths, graph.node_list),
        )

    @concrete_algorithm("traversal.minimum_spanning_tree")
    def ss_minimum_spanning_tree(graph: ScipyGraph) -> ScipyGraph:
        span_tree = ss.csgraph.minimum_spanning_tree(graph.value)
        span_tree_mask = (span_tree != 0).astype(int, copy=False)
        span_tree_mask_transposed = span_tree_mask.T
        divisor = span_tree_mask + span_tree_mask_transposed
        undirected_span_tree = span_tree + span_tree.T
        # assert (undirected_span_tree.indices == divisor.indices).all()
        # assert (undirected_span_tree.indptr == divisor.indptr).all()
        undirected_span_tree.data /= divisor.data
        undirected_span_tree = undirected_span_tree.astype(
            graph.value.dtype, copy=False
        )
        return ScipyGraph(undirected_span_tree, graph.node_list, graph.node_vals)

    @concrete_algorithm("cluster.triangle_count")
    def ss_triangle_count(graph: ScipyGraph) -> int:
        """
        Uses the triangle counting method described in
        https://www.sandia.gov/~srajama/publications/Tricount-HPEC.pdf
        """
        props = ScipyGraph.Type.compute_abstract_properties(graph, {"edge_type"})
        if props["edge_type"] == "map":
            # Drop weights before performing triangle count
            m = graph.value.copy()
            m.data = np.ones_like(m.data)
        elif props["edge_type"] == "set":
            m = graph.value
        L = ss.tril(m, k=-1).tocsr()
        U = ss.triu(m, k=1).tocsc()
        return int((L @ U.T).multiply(L).sum())

    @concrete_algorithm("traversal.bfs_iter")
    def ss_breadth_first_search_iter(
        graph: ScipyGraph, source_node: NodeID, depth_limit: int
    ) -> NumpyVectorType:
        is_directed = ScipyGraph.Type.compute_abstract_properties(
            graph, {"is_directed"}
        )["is_directed"]
        bfs_ordered_incides = ss.csgraph.breadth_first_order(
            graph.value, source_node, directed=is_directed, return_predecessors=False,
        )
        bfs_ordered_nodes = graph.node_list[bfs_ordered_incides]
        return bfs_ordered_nodes

    @concrete_algorithm("flow.max_flow")
    def ss_max_flow(
        graph: ScipyGraph, source_node: NodeID, target_node: NodeID,
    ) -> Tuple[float, ScipyGraph]:
        max_flow_result = ss.csgraph.maximum_flow(graph.value, source_node, target_node)
        flow_value = max_flow_result.flow_value
        residual_graph = max_flow_result.residual
        residual_keep_mask = residual_graph > 0
        ss_flow_graph = residual_graph.multiply(residual_keep_mask)
        flow_graph = ScipyGraph(
            ss_flow_graph, node_list=graph.node_list, node_vals=graph.node_vals
        )
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
        if not isinstance(func, np.ufunc):
            func = np.frompyfunc(func, 2, 1)
        nrows = graph.value.shape[0]
        agg_values = np.full(nrows, initial_value)
        if in_edges:
            csc_matrix = graph.value.tocsc()
            in_edges_aggregated_values, keep_mask = _reduce_sparse_matrix(
                func, csc_matrix
            )
            agg_values[keep_mask] = func(
                agg_values[keep_mask], in_edges_aggregated_values
            )
        if out_edges:
            csr_matrix = graph.value.tocsr()
            out_edges_aggregated_values, keep_mask = _reduce_sparse_matrix(
                func, csr_matrix
            )
            agg_values[keep_mask] = func(
                agg_values[keep_mask], out_edges_aggregated_values
            )

        return NumpyNodeMap(agg_values, nodes=graph.node_list)

    @concrete_algorithm("util.graph.filter_edges")
    def ss_graph_filter_edges(
        graph: ScipyGraph, func: Callable[[Any], bool]
    ) -> ScipyGraph:
        # TODO consider caching this somewhere or enforcing that only vectorized functions are given
        func_vectorized = numba.vectorize(func) if has_numba else np.vectorize(func)
        # TODO Explicitly handle the CSR case
        result_matrix = graph.value.tocoo(copy=True)
        to_keep_mask = func_vectorized(result_matrix.data)
        if not to_keep_mask.all():
            result_matrix.row = result_matrix.row[to_keep_mask]
            result_matrix.col = result_matrix.col[to_keep_mask]
            result_matrix.data = result_matrix.data[to_keep_mask]
        return ScipyGraph(result_matrix, graph.node_list, graph.node_vals)

    @concrete_algorithm("util.graph.assign_uniform_weight")
    def ss_graph_assign_uniform_weight(graph: ScipyGraph, weight: Any) -> ScipyGraph:
        matrix = graph.value.copy()
        matrix.data.fill(weight)
        return ScipyGraph(matrix, graph.node_list, graph.node_vals)

    @concrete_algorithm("util.graph.build")
    def ss_graph_build(
        edges: Union[ScipyEdgeSet, ScipyEdgeMap],
        nodes: Union[NumpyNodeSet, NumpyNodeMap, None],
    ) -> ScipyGraph:
        aprops = {
            "edge_type": "map" if isinstance(edges, ScipyEdgeMap) else "set",
            "node_type": "map" if isinstance(nodes, NumpyNodeMap) else "set",
        }
        m = edges.value.copy()
        node_list = edges.node_list.copy()
        node_vals = None
        if nodes is not None:
            all_nodes = nodes.nodes if aprops["node_type"] == "map" else nodes.value
            isolates = np.setdiff1d(all_nodes, node_list)
            if len(isolates) > 0:
                new_size = m.shape[0] + len(isolates)
                m.resize((new_size, new_size))
                node_list = np.concatenate([node_list, isolates])
            if aprops["node_type"] == "map":
                node_vals = nodes.value.copy()
                if len(isolates) > 0:
                    # align ordering of node values
                    sorter = np.argsort(node_list)
                    node_vals[sorter] = nodes.value
        return ScipyGraph(m, node_list, node_vals, aprops=aprops)

    @concrete_algorithm("util.edgemap.from_edgeset")
    def ss_edgemap_from_edgeset(
        edgeset: ScipyEdgeSet, default_value: Any,
    ) -> ScipyEdgeMap:
        new_matrix = edgeset.value.copy()
        new_matrix.data.fill(default_value)
        return ScipyEdgeMap(new_matrix, edgeset.node_list.copy())
