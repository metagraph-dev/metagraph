from metagraph import concrete_algorithm
from metagraph.plugins import has_networkx, has_community
from typing import Tuple, Iterable, Any


if has_networkx:
    import networkx as nx
    import numpy as np
    from .types import NetworkXGraph
    from ..python.types import PythonNodes
    from ..numpy.types import NumpyVector

    @concrete_algorithm("link_analysis.pagerank")
    def nx_pagerank(
        graph: NetworkXGraph, damping: float, maxiter: int, tolerance: float
    ) -> PythonNodes:
        pagerank = nx.pagerank(
            graph.value, alpha=damping, max_iter=maxiter, tol=tolerance, weight=None
        )
        return PythonNodes(
            pagerank, dtype="float", weights="positive", node_index=graph.node_index
        )

    @concrete_algorithm("link_analysis.katz_centrality")
    def nx_katz_centrality(
        graph: NetworkXGraph,
        attenuation_factor: float,
        immediate_neighbor_weight: float,
        maxiter: int,
        tolerance: float,
    ) -> PythonNodes:
        katz_centrality_scores = nx.katz_centrality(
            graph.value,
            alpha=attenuation_factor,
            beta=immediate_neighbor_weight,
            max_iter=maxiter,
            tol=tolerance,
            weight=None,
        )
        return PythonNodes(
            katz_centrality_scores,
            dtype="float",
            weights="positive",
            node_index=graph.node_index,
        )

    @concrete_algorithm("cluster.triangle_count")
    def nx_triangle_count(graph: NetworkXGraph) -> int:
        triangles = nx.triangles(graph.value)
        # Sum up triangles from each node
        # Divide by 3 because each triangle is counted 3 times
        total_triangles = sum(triangles.values()) // 3
        return total_triangles

    @concrete_algorithm("clustering.connected_components")
    def nx_connected_components(graph: NetworkXGraph) -> PythonNodes:
        index_to_label = dict()
        for i, nodes in enumerate(nx.connected_components(graph.value)):
            for node in nodes:
                index_to_label[node] = i
        return PythonNodes(
            index_to_label,
            node_index=graph.node_index,
            dtype="int",
            weights="non-negative",
        )

    @concrete_algorithm("clustering.strongly_connected_components")
    def nx_strongly_connected_components(graph: NetworkXGraph) -> PythonNodes:
        index_to_label = dict()
        for i, nodes in enumerate(nx.strongly_connected_components(graph.value)):
            for node in nodes:
                index_to_label[node] = i
        return PythonNodes(
            index_to_label,
            node_index=graph.node_index,
            dtype="int",
            weights="non-negative",
        )

    @concrete_algorithm("clustering.label_propagation_community")
    def nx_label_propagation_community(graph: NetworkXGraph) -> PythonNodes:
        communities = nx.algorithms.community.label_propagation.label_propagation_communities(
            graph.value
        )
        index_to_label = dict()
        for label, nodes in enumerate(communities):
            for node in nodes:
                index_to_label[node] = label
        return PythonNodes(
            index_to_label,
            node_index=graph.node_index,
            dtype="int",
            weights="non-negative",
        )

    @concrete_algorithm("subgraph.extract_subgraph")
    def nx_extract_subgraph(
        graph: NetworkXGraph, nodes: Iterable[Any]
    ) -> NetworkXGraph:
        subgraph = graph.value.subgraph(nodes)
        return NetworkXGraph(subgraph, weight_label=graph.weight_label,)

    @concrete_algorithm("subgraph.k_core")
    def nx_k_core(graph: NetworkXGraph, k: int) -> NetworkXGraph:
        k_core_graph = nx.k_core(graph.value, k)
        return NetworkXGraph(k_core_graph, weight_label=graph.weight_label,)

    @concrete_algorithm("traversal.bellman_ford")
    def nx_bellman_ford(
        graph: NetworkXGraph, source_node: Any
    ) -> Tuple[PythonNodes, PythonNodes]:
        predecessors_map, distance_map = nx.bellman_ford_predecessor_and_distance(
            graph.value, source_node
        )
        single_parent_map = {
            child: parents[0] if len(parents) > 0 else source_node
            for child, parents in predecessors_map.items()
        }
        return (
            PythonNodes(
                single_parent_map,
                node_index=graph.node_index,
                dtype="int",
                weights="non-negative",
            ),
            PythonNodes(
                distance_map,
                node_index=graph.node_index,
                dtype="float",
                weights="non-negative",
            ),
        )

    @concrete_algorithm("traversal.dijkstra")
    def dijkstra(
        graph: NetworkXGraph, source_node: Any, max_path_length: float
    ) -> Tuple[PythonNodes, PythonNodes]:
        predecessors_map, distance_map = nx.dijkstra_predecessor_and_distance(
            graph.value, source_node, cutoff=max_path_length,
        )
        single_parent_map = {
            child: parents[0] if len(parents) > 0 else source_node
            for child, parents in predecessors_map.items()
        }
        return (
            PythonNodes(
                single_parent_map,
                node_index=graph.node_index,
                dtype="int",
                weights="non-negative",
            ),
            PythonNodes(
                distance_map,
                node_index=graph.node_index,
                dtype="float",
                weights="non-negative",
            ),
        )

    @concrete_algorithm("vertex_ranking.betweenness_centrality")
    def nx_betweenness_centrality(
        graph: NetworkXGraph,
        k: int,
        enable_normalization: bool,
        include_endpoints: bool,
    ) -> PythonNodes:
        node_to_score_map = nx.betweenness_centrality(
            graph.value, k, enable_normalization, include_endpoints
        )
        return PythonNodes(
            node_to_score_map,
            node_index=graph.node_index,
            dtype="float",
            weights="non-negative",
        )

    @concrete_algorithm("traversal.breadth_first_search")
    def nx_breadth_first_search(graph: NetworkXGraph, source_node: Any) -> NumpyVector:
        bfs_ordered_node_array = np.array(
            nx.breadth_first_search.bfs_tree(graph.value, source_node)
        )
        return NumpyVector(bfs_ordered_node_array)


if has_community:
    import community as community_louvain
    from .types import NetworkXGraph
    from ..python.types import PythonNodes

    @concrete_algorithm("clustering.louvain_community")
    def nx_louvain_community(graph: NetworkXGraph) -> Tuple[PythonNodes, float]:
        index_to_label = community_louvain.best_partition(graph.value)
        modularity_score = community_louvain.modularity(index_to_label, graph.value)
        return (
            PythonNodes(
                index_to_label,
                node_index=graph.node_index,
                dtype="int",
                weights="non-negative",
            ),
            modularity_score,
        )
