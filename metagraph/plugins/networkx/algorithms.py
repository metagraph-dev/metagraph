import metagraph as mg
from metagraph import concrete_algorithm, NodeID
from metagraph.plugins import has_networkx, has_community, has_pandas
from typing import Tuple, Iterable, Any, Callable


if has_networkx:
    import networkx as nx
    import numpy as np
    from .types import NetworkXGraph, NetworkXBipartiteGraph
    from ..python.types import PythonNodeMap, PythonNodeSet
    from ..numpy.types import NumpyVector

    @concrete_algorithm("centrality.pagerank")
    def nx_pagerank(
        graph: NetworkXGraph, damping: float, maxiter: int, tolerance: float
    ) -> PythonNodeMap:
        pagerank = nx.pagerank(
            graph.value, alpha=damping, max_iter=maxiter, tol=tolerance, weight=None
        )
        return PythonNodeMap(pagerank)

    @concrete_algorithm("centrality.katz")
    def nx_katz_centrality(
        graph: NetworkXGraph,
        attenuation_factor: float,
        immediate_neighbor_weight: float,
        maxiter: int,
        tolerance: float,
    ) -> PythonNodeMap:
        katz_centrality_scores = nx.katz_centrality(
            graph.value,
            alpha=attenuation_factor,
            beta=immediate_neighbor_weight,
            max_iter=maxiter,
            tol=tolerance,
            weight=graph.edge_weight_label,
        )
        return PythonNodeMap(katz_centrality_scores)

    @concrete_algorithm("cluster.triangle_count")
    def nx_triangle_count(graph: NetworkXGraph) -> int:
        triangles = nx.triangles(graph.value)
        # Sum up triangles from each node
        # Divide by 3 because each triangle is counted 3 times
        total_triangles = sum(triangles.values()) // 3
        return total_triangles

    @concrete_algorithm("clustering.connected_components")
    def nx_connected_components(graph: NetworkXGraph) -> PythonNodeMap:
        index_to_label = dict()
        for i, nodes in enumerate(nx.connected_components(graph.value)):
            for node in nodes:
                index_to_label[node] = i
        return PythonNodeMap(index_to_label,)

    @concrete_algorithm("clustering.strongly_connected_components")
    def nx_strongly_connected_components(graph: NetworkXGraph) -> PythonNodeMap:
        index_to_label = dict()
        for i, nodes in enumerate(nx.strongly_connected_components(graph.value)):
            for node in nodes:
                index_to_label[node] = i
        return PythonNodeMap(index_to_label,)

    @concrete_algorithm("clustering.label_propagation_community")
    def nx_label_propagation_community(graph: NetworkXGraph) -> PythonNodeMap:
        communities = nx.algorithms.community.label_propagation.label_propagation_communities(
            graph.value
        )
        index_to_label = dict()
        for label, nodes in enumerate(communities):
            for node in nodes:
                index_to_label[node] = label
        return PythonNodeMap(index_to_label,)

    @concrete_algorithm("subgraph.extract_subgraph")
    def nx_extract_subgraph(
        graph: NetworkXGraph, nodes: PythonNodeSet
    ) -> NetworkXGraph:
        subgraph = graph.value.subgraph(nodes.value)
        return NetworkXGraph(subgraph, edge_weight_label=graph.edge_weight_label)

    @concrete_algorithm("subgraph.k_core")
    def nx_k_core(graph: NetworkXGraph, k: int) -> NetworkXGraph:
        k_core_graph = nx.k_core(graph.value, k)
        return NetworkXGraph(k_core_graph, edge_weight_label=graph.edge_weight_label)

    @concrete_algorithm("traversal.bellman_ford")
    def nx_bellman_ford(
        graph: NetworkXGraph, source_node: NodeID
    ) -> Tuple[PythonNodeMap, PythonNodeMap]:
        predecessors_map, distance_map = nx.bellman_ford_predecessor_and_distance(
            graph.value, source_node
        )
        single_parent_map = {
            child: parents[0] if len(parents) > 0 else source_node
            for child, parents in predecessors_map.items()
        }
        return (
            PythonNodeMap(single_parent_map,),
            PythonNodeMap(distance_map,),
        )

    @concrete_algorithm("traversal.dijkstra")
    def nx_dijkstra(
        graph: NetworkXGraph, source_node: NodeID  # , max_path_length: float
    ) -> Tuple[PythonNodeMap, PythonNodeMap]:
        predecessors_map, distance_map = nx.dijkstra_predecessor_and_distance(
            graph.value, source_node,  # cutoff=max_path_length,
        )
        single_parent_map = {
            child: parents[0] if len(parents) > 0 else source_node
            for child, parents in predecessors_map.items()
        }
        return (
            PythonNodeMap(single_parent_map,),
            PythonNodeMap(distance_map,),
        )

    @concrete_algorithm("centrality.betweenness")
    def nx_betweenness_centrality(
        graph: NetworkXGraph,
        nodes: NumpyVector,
        normalize: bool,
        # include_endpoints: bool,
    ) -> PythonNodeMap:
        node_to_score_map = nx.betweenness_centrality_subset(
            graph.value,
            sources=nodes.value,
            targets=nodes.value,
            normalized=normalize,
            weight=graph.edge_weight_label,
            # endpoints=include_endpoints,
        )
        return PythonNodeMap(node_to_score_map,)

    @concrete_algorithm("traversal.bfs_iter")
    def nx_breadth_first_search(
        graph: NetworkXGraph, source_node: NodeID, depth_limit: int
    ) -> NumpyVector:
        bfs_ordered_node_array = np.array(
            nx.breadth_first_search.bfs_tree(graph.value, source_node)
        )
        return NumpyVector(bfs_ordered_node_array)

    @concrete_algorithm("bipartite.graph_projection")
    def nx_graph_projection(
        bgraph: NetworkXBipartiteGraph, nodes_retained: int
    ) -> NetworkXGraph:
        g_proj = nx.projected_graph(bgraph.value, bgraph.nodes[nodes_retained])
        return NetworkXGraph(
            g_proj,
            node_weight_label=bgraph.node_weight_label,
            edge_weight_label=bgraph.edge_weight_label,
        )

    @concrete_algorithm("util.graph.aggregate_edges")
    def nx_graph_aggregate_edges(
        graph: NetworkXGraph,
        func: Callable[[Any, Any], Any],
        initial_value: Any,
        in_edges: bool,
        out_edges: bool,
    ) -> PythonNodeMap:
        if not (in_edges == out_edges):
            is_directed = NetworkXGraph.Type.compute_abstract_properties(
                graph, {"is_directed"}
            )["is_directed"]
            if not is_directed:
                in_edges = out_edges = in_edges or out_edges
        result_dict = {node: initial_value for node in graph.value.nodes}
        for start_node, end_node, weight in graph.value.edges.data(
            graph.edge_weight_label
        ):
            if out_edges:
                result_dict[start_node] = func(weight, result_dict[start_node])
            if in_edges:
                result_dict[end_node] = func(weight, result_dict[end_node])
        return PythonNodeMap(result_dict)

    @concrete_algorithm("util.graph.filter_edges")
    def nx_graph_filter_edges(
        graph: NetworkXGraph, func: Callable[[Any], bool]
    ) -> NetworkXGraph:
        result_nx_graph = type(graph.value)()
        result_nx_graph.add_nodes_from(graph.value.nodes())
        ebunch = filter(
            lambda uvw_triple: func(uvw_triple[-1]),
            graph.value.edges.data(graph.edge_weight_label),
        )
        result_nx_graph.add_weighted_edges_from(ebunch)
        return NetworkXGraph(
            result_nx_graph,
            node_weight_label=graph.node_weight_label,
            edge_weight_label=graph.edge_weight_label,
        )

    @concrete_algorithm("util.graph.add_uniform_weight")
    def nx_graph_add_uniform_weight(graph: NetworkXGraph, weight: Any) -> NetworkXGraph:
        result_nx_graph = graph.value.copy()
        for _, _, edge_attributes in result_nx_graph.edges.data():
            edge_attributes[graph.edge_weight_label] += weight
        return NetworkXGraph(
            result_nx_graph, graph.node_weight_label, graph.edge_weight_label
        )


if has_networkx and has_community:
    import community as community_louvain
    from .types import NetworkXGraph
    from ..python.types import PythonNodeMap

    @concrete_algorithm("clustering.louvain_community")
    def nx_louvain_community(graph: NetworkXGraph) -> Tuple[PythonNodeMap, float]:
        index_to_label = community_louvain.best_partition(graph.value)
        modularity_score = community_louvain.modularity(index_to_label, graph.value)
        return (
            PythonNodeMap(index_to_label,),
            modularity_score,
        )


if has_networkx and has_pandas:
    from ..pandas.types import PandasEdgeSet, PandasEdgeMap
    from ..python.types import PythonNodeMap, PythonNodeSet

    @concrete_algorithm("util.graph.build")
    def nx_graph_build_from_pandas(
        edges: mg.Union[PandasEdgeSet, PandasEdgeMap],
        nodes: mg.Optional[mg.Union[PythonNodeSet, PythonNodeMap]],
    ) -> NetworkXGraph:
        g = nx.DiGraph() if edges.is_directed else nx.Graph()
        if nodes is not None:
            if type(nodes) is PythonNodeMap:
                g.add_nodes_from((n, {"weight": v}) for n, v in nodes.value.items())
            else:
                g.add_nodes_from(nodes.value)
        if type(edges) is PandasEdgeMap:
            df = edges.value[[edges.src_label, edges.dst_label, edges.weight_label]]
            g.add_weighted_edges_from(df.itertuples(index=False, name="WeightedEdge"))
        else:
            df = edges.value[[edges.src_label, edges.dst_label]]
            g.add_edges_from(df.itertuples(index=False, name="Edge"))
        return NetworkXGraph(g)
