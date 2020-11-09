import metagraph as mg
from metagraph import concrete_algorithm, NodeID
from metagraph.plugins import has_networkx, has_community, has_pandas
from typing import Tuple, Any, Callable
import random


if has_networkx:
    import networkx as nx
    import numpy as np
    from .types import NetworkXGraph, NetworkXBipartiteGraph
    from ..python.types import PythonNodeMapType, PythonNodeSetType
    from ..numpy.types import NumpyVectorType

    @concrete_algorithm("centrality.pagerank")
    def nx_pagerank(
        graph: NetworkXGraph, damping: float, maxiter: int, tolerance: float
    ) -> PythonNodeMapType:
        pagerank = nx.pagerank(
            graph.value, alpha=damping, max_iter=maxiter, tol=tolerance, weight=None
        )
        return pagerank

    @concrete_algorithm("centrality.katz")
    def nx_katz_centrality(
        graph: NetworkXGraph,
        attenuation_factor: float,
        immediate_neighbor_weight: float,
        maxiter: int,
        tolerance: float,
    ) -> PythonNodeMapType:
        katz_centrality_scores = nx.katz_centrality(
            graph.value,
            alpha=attenuation_factor,
            beta=immediate_neighbor_weight,
            max_iter=maxiter,
            tol=tolerance,
            weight=graph.edge_weight_label,
        )
        return katz_centrality_scores

    @concrete_algorithm("cluster.triangle_count")
    def nx_triangle_count(graph: NetworkXGraph) -> int:
        triangles = nx.triangles(graph.value)
        # Sum up triangles from each node
        # Divide by 3 because each triangle is counted 3 times
        total_triangles = sum(triangles.values()) // 3
        return total_triangles

    @concrete_algorithm("clustering.connected_components")
    def nx_connected_components(graph: NetworkXGraph) -> PythonNodeMapType:
        index_to_label = {}
        for i, nodes in enumerate(nx.connected_components(graph.value)):
            for node in nodes:
                index_to_label[node] = i
        return index_to_label

    @concrete_algorithm("clustering.strongly_connected_components")
    def nx_strongly_connected_components(graph: NetworkXGraph) -> PythonNodeMapType:
        index_to_label = {}
        for i, nodes in enumerate(nx.strongly_connected_components(graph.value)):
            for node in nodes:
                index_to_label[node] = i
        return index_to_label

    @concrete_algorithm("clustering.label_propagation_community")
    def nx_label_propagation_community(graph: NetworkXGraph) -> PythonNodeMapType:
        communities = nx.algorithms.community.label_propagation.label_propagation_communities(
            graph.value
        )
        index_to_label = {}
        for label, nodes in enumerate(communities):
            for node in nodes:
                index_to_label[node] = label
        return index_to_label

    @concrete_algorithm("subgraph.extract_subgraph")
    def nx_extract_subgraph(
        graph: NetworkXGraph, nodes: PythonNodeSetType
    ) -> NetworkXGraph:
        subgraph = graph.value.subgraph(nodes)
        return NetworkXGraph(subgraph, edge_weight_label=graph.edge_weight_label)

    @concrete_algorithm("subgraph.k_core")
    def nx_k_core(graph: NetworkXGraph, k: int) -> NetworkXGraph:
        k_core_graph = nx.k_core(graph.value, k)
        return NetworkXGraph(
            k_core_graph,
            node_weight_label=graph.node_weight_label,
            edge_weight_label=graph.edge_weight_label,
        )

    if nx.__version__ >= "2.4":

        @concrete_algorithm("subgraph.k_truss")
        def nx_k_truss(graph: NetworkXGraph, k: int) -> NetworkXGraph:
            if nx.__version__ < "2.5":
                # v2.4 uses `k` rather than `k-2` as everyone else uses
                k -= 2
            k_truss_graph = nx.k_truss(graph.value, k)
            return NetworkXGraph(
                k_truss_graph,
                node_weight_label=graph.node_weight_label,
                edge_weight_label=graph.edge_weight_label,
            )

    @concrete_algorithm("subgraph.maximal_independent_set")
    def maximal_independent_set(graph: NetworkXGraph) -> PythonNodeSetType:
        nodes = nx.maximal_independent_set(graph.value)
        return set(nodes)

    @concrete_algorithm("traversal.bellman_ford")
    def nx_bellman_ford(
        graph: NetworkXGraph, source_node: NodeID
    ) -> Tuple[PythonNodeMapType, PythonNodeMapType]:
        predecessors_map, distance_map = nx.bellman_ford_predecessor_and_distance(
            graph.value, source_node
        )
        single_parent_map = {
            child: parents[0] if len(parents) > 0 else source_node
            for child, parents in predecessors_map.items()
        }
        return (single_parent_map, distance_map)

    @concrete_algorithm("traversal.dijkstra")
    def nx_dijkstra(
        graph: NetworkXGraph, source_node: NodeID  # , max_path_length: float
    ) -> Tuple[PythonNodeMapType, PythonNodeMapType]:
        predecessors_map, distance_map = nx.dijkstra_predecessor_and_distance(
            graph.value, source_node,  # cutoff=max_path_length,
        )
        single_parent_map = {
            child: parents[0] if len(parents) > 0 else source_node
            for child, parents in predecessors_map.items()
        }
        return (single_parent_map, distance_map)

    @concrete_algorithm("traversal.minimum_spanning_tree")
    def nx_minimum_spanning_tree(graph: NetworkXGraph) -> NetworkXGraph:
        mst_graph = nx.minimum_spanning_tree(graph.value)
        return NetworkXGraph(
            mst_graph,
            node_weight_label=graph.node_weight_label,
            edge_weight_label=graph.edge_weight_label,
        )

    @concrete_algorithm("centrality.betweenness")
    def nx_betweenness_centrality(
        graph: NetworkXGraph, nodes: mg.Optional[PythonNodeSetType], normalize: bool,
    ) -> PythonNodeMapType:
        if nodes is None:
            sources = targets = graph.value.nodes
        else:
            sources = targets = nodes
        node_to_score_map = nx.betweenness_centrality_subset(
            graph.value,
            sources=sources,
            targets=targets,
            normalized=normalize,
            weight=graph.edge_weight_label,
        )
        return node_to_score_map

    @concrete_algorithm("centrality.closeness")
    def nx_closeness_centrality(
        graph: NetworkXGraph, nodes: mg.Optional[PythonNodeSetType],
    ) -> PythonNodeMapType:
        if nodes is None:
            result = nx.closeness_centrality(
                graph.value, distance=graph.edge_weight_label
            )
        else:
            result = {
                node: nx.closeness_centrality(
                    graph.value, node, distance=graph.edge_weight_label
                )
                for node in nodes
            }
        return result

    @concrete_algorithm("centrality.eigenvector")
    def nx_eigenvector_centrality(
        graph: NetworkXGraph, maxiter: int, tolerance: float
    ) -> PythonNodeMapType:
        result = nx.eigenvector_centrality(
            graph.value, maxiter, tolerance, weight=graph.edge_weight_label
        )
        return result

    @concrete_algorithm("centrality.hits")
    def nx_hits_centrality(
        graph: NetworkXGraph, maxiter: int, tolerance: float, normalize: bool,
    ) -> Tuple[PythonNodeMapType, PythonNodeMapType]:
        hubs, authority = nx.hits(graph.value, maxiter, tolerance, normalized=normalize)
        return hubs, authority

    @concrete_algorithm("traversal.bfs_iter")
    def nx_breadth_first_search(
        graph: NetworkXGraph, source_node: NodeID, depth_limit: int
    ) -> NumpyVectorType:
        bfs_ordered_node_array = np.array(
            nx.breadth_first_search.bfs_tree(graph.value, source_node)
        )
        return bfs_ordered_node_array

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

    @concrete_algorithm("flow.max_flow")
    def nx_max_flow(
        graph: NetworkXGraph, source_node: NodeID, target_node: NodeID,
    ) -> Tuple[float, NetworkXGraph]:
        flow_value, flow_dict = nx.maximum_flow(
            graph.value, source_node, target_node, capacity=graph.edge_weight_label
        )
        nx_flow_graph = nx.DiGraph()
        for src in flow_dict.keys():
            for dst in flow_dict[src].keys():
                nx_flow_graph.add_edge(
                    src, dst, **{graph.edge_weight_label: flow_dict[src][dst]}
                )
        flow_graph = NetworkXGraph(
            nx_flow_graph,
            node_weight_label=graph.node_weight_label,
            edge_weight_label=graph.edge_weight_label,
        )
        return (flow_value, flow_graph)

    @concrete_algorithm("flow.min_cut")
    def nx_min_cut(
        graph: NetworkXGraph, source_node: NodeID, target_node: NodeID,
    ) -> Tuple[float, NetworkXGraph]:
        g = graph.value
        flow_value, (reachable, non_reachable) = nx.minimum_cut(
            g, source_node, target_node, capacity=graph.edge_weight_label
        )
        # Build graph containing cut edges
        nx_cut_graph = type(g)()
        nx_cut_graph.add_nodes_from(g.nodes(data=True))
        for u, nbrs in ((n, g[n]) for n in reachable):
            for v in nbrs:
                if v in non_reachable:
                    edge_attrs = g.edges[u, v]
                    nx_cut_graph.add_edge(u, v, **edge_attrs)
        cut_graph = NetworkXGraph(
            nx_cut_graph,
            node_weight_label=graph.node_weight_label,
            edge_weight_label=graph.edge_weight_label,
        )
        return flow_value, cut_graph

    @concrete_algorithm("util.graph.degree")
    def nx_graph_degree(
        graph: NetworkXGraph, in_edges: bool, out_edges: bool
    ) -> PythonNodeMapType:
        if in_edges and out_edges:
            ins = graph.value.in_degree()
            outs = graph.value.out_degree()
            d = {n: ins[n] + o for n, o in outs}
        elif in_edges:
            d = dict(graph.value.in_degree())
        elif out_edges:
            d = dict(graph.value.out_degree())
        else:
            d = {n: 0 for n in graph.value.nodes()}
        return d

    @concrete_algorithm("util.graph.aggregate_edges")
    def nx_graph_aggregate_edges(
        graph: NetworkXGraph,
        func: Callable[[Any, Any], Any],
        initial_value: Any,
        in_edges: bool,
        out_edges: bool,
    ) -> PythonNodeMapType:
        result_dict = {node: initial_value for node in graph.value.nodes}
        if in_edges or out_edges:
            if in_edges != out_edges:
                is_directed = NetworkXGraph.Type.compute_abstract_properties(
                    graph, {"is_directed"}
                )["is_directed"]
                if not is_directed:
                    in_edges = out_edges = True
            for start_node, end_node, weight in graph.value.edges.data(
                graph.edge_weight_label
            ):
                if out_edges:
                    result_dict[start_node] = func(weight, result_dict[start_node])
                if in_edges:
                    result_dict[end_node] = func(weight, result_dict[end_node])
        return result_dict

    @concrete_algorithm("util.graph.filter_edges")
    def nx_graph_filter_edges(
        graph: NetworkXGraph, func: Callable[[Any], bool]
    ) -> NetworkXGraph:
        result_nx_graph = type(graph.value)()
        result_nx_graph.add_nodes_from(graph.value.nodes.data())
        ebunch = filter(
            lambda uvw_triple: func(uvw_triple[-1]),
            graph.value.edges.data(data=graph.edge_weight_label),
        )
        result_nx_graph.add_weighted_edges_from(ebunch, weight=graph.edge_weight_label)
        return NetworkXGraph(
            result_nx_graph,
            node_weight_label=graph.node_weight_label,
            edge_weight_label=graph.edge_weight_label,
        )

    @concrete_algorithm("util.graph.assign_uniform_weight")
    def nx_graph_assign_uniform_weight(
        graph: NetworkXGraph, weight: Any
    ) -> NetworkXGraph:
        result_nx_graph = graph.value.copy()
        for _, _, edge_attributes in result_nx_graph.edges.data():
            edge_attributes[graph.edge_weight_label] = weight
        return NetworkXGraph(
            result_nx_graph, graph.node_weight_label, graph.edge_weight_label
        )

    @concrete_algorithm("clustering.coloring.greedy")
    def nx_greedy_coloring(graph: NetworkXGraph) -> Tuple[PythonNodeMapType, int]:
        colors = nx.greedy_color(graph.value)
        unique_colors = set(colors.values())
        return colors, len(unique_colors)

    @concrete_algorithm("subgraph.sample.node_sampling")
    def nx_node_sampling(graph: NetworkXGraph, p: float) -> NetworkXGraph:
        if p <= 0 or p > 1:
            raise ValueError(f"Probability `p` must be between 0 and 1, found {p}")
        aprops = NetworkXGraph.Type.compute_abstract_properties(
            graph, {"node_type", "edge_type"}
        )
        g = graph.value
        ns = set(n for n in g.nodes() if random.random() < p)
        subgraph = type(g)()
        if aprops["node_type"] == "map":
            for n in ns:
                subgraph.add_node(n, **g.nodes[n])
        else:
            subgraph.add_nodes_from(ns)
        # Add edges which exist between selected nodes
        if aprops["edge_type"] == "map":
            for n in ns:
                for nbr in set(g.neighbors(n)) & ns:
                    subgraph.add_edge(n, nbr, **g[n][nbr])
        else:
            for n in ns:
                for nbr in set(g.neighbors(n)) & ns:
                    subgraph.add_edge(n, nbr)
        return NetworkXGraph(
            subgraph,
            node_weight_label=graph.node_weight_label,
            edge_weight_label=graph.edge_weight_label,
        )

    @concrete_algorithm("subgraph.sample.edge_sampling")
    def nx_edge_sampling(graph: NetworkXGraph, p: float) -> NetworkXGraph:
        if p <= 0 or p > 1:
            raise ValueError(f"Probability `p` must be between 0 and 1, found {p}")
        aprops = NetworkXGraph.Type.compute_abstract_properties(
            graph, {"node_type", "edge_type"}
        )
        g = graph.value
        es = set(e for e in g.edges() if random.random() < p)
        ns = set(src for src, dst in es) | set(dst for src, dst in es)
        subgraph = type(g)()
        if aprops["node_type"] == "map":
            for n in ns:
                subgraph.add_node(n, **g.nodes[n])
        else:
            subgraph.add_nodes_from(ns)
        # Add edges which exist between selected nodes
        if aprops["edge_type"] == "map":
            for e in es:
                subgraph.add_edge(*e, **g.edges[e])
        else:
            subgraph.add_edges_from(es)
        return NetworkXGraph(
            subgraph,
            node_weight_label=graph.node_weight_label,
            edge_weight_label=graph.edge_weight_label,
        )

    @concrete_algorithm("subgraph.sample.ties")
    def nx_ties(graph: NetworkXGraph, p: float) -> NetworkXGraph:
        """
        Totally Induced Edge Sampling method
        https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2743&context=cstech
        """
        if p <= 0 or p > 1:
            raise ValueError(f"Probability `p` must be between 0 and 1, found {p}")
        aprops = NetworkXGraph.Type.compute_abstract_properties(
            graph, {"node_type", "edge_type"}
        )
        g = graph.value
        es = set(e for e in g.edges() if random.random() < p)
        ns = set(src for src, dst in es) | set(dst for src, dst in es)
        subgraph = type(g)()
        if aprops["node_type"] == "map":
            for n in ns:
                subgraph.add_node(n, **g.nodes[n])
        else:
            subgraph.add_nodes_from(ns)
        # Add edges which exist between selected nodes
        if aprops["edge_type"] == "map":
            for n in ns:
                for nbr in set(g.neighbors(n)) & ns:
                    subgraph.add_edge(n, nbr, **g[n][nbr])
        else:
            for n in ns:
                for nbr in set(g.neighbors(n)) & ns:
                    subgraph.add_edge(n, nbr)
        return NetworkXGraph(
            subgraph,
            node_weight_label=graph.node_weight_label,
            edge_weight_label=graph.edge_weight_label,
        )

    @concrete_algorithm("subgraph.sample.random_walk")
    def nx_random_walk_sampling(
        graph: NetworkXGraph,
        num_steps: mg.Optional[int],
        num_nodes: mg.Optional[int],
        num_edges: mg.Optional[int],
        jump_probability: float,
        start_node: mg.Optional[NodeID],
    ) -> NetworkXGraph:
        """
        Sample using random walks

        Sampling ends when number of steps, nodes, or edges are reached (first to occur if multiple are specified).
        For each step, there is a jump_probability to reset the walk.
        When resetting the walk, if start_node is specified, always reset to this node. If not specified, every reset
            picks a new node in the graph at random.
        """
        if jump_probability <= 0 or jump_probability > 1:
            raise ValueError(
                f"`jump_probability` must be between 0 and 1, found {jump_probability}"
            )
        if num_steps is None and num_nodes is None and num_edges is None:
            raise ValueError(
                "Must specify at least one of num_steps, num_nodes, or num_edges"
            )

        g = graph.value
        out_g = type(g)()
        if start_node is None:
            possible_nodes = list(g.nodes())
            current_node = random.choice(possible_nodes)
        else:
            current_node = start_node
            if len(g[current_node]) == 0:
                # We always start with this, and there are no out edges, so we will never finish
                out_g.add_node(current_node, **g.nodes[current_node])
                return NetworkXGraph(out_g)

        out_g.add_node(current_node, **g.nodes[current_node])

        counter = 0
        while True:
            counter += 1

            nbrs = list(g[current_node])
            if len(nbrs) > 0 and random.random() >= jump_probability:
                prev_node = current_node
                current_node = random.choice(nbrs)
                out_g.add_node(current_node, **g.nodes[current_node])
                out_g.add_edge(
                    prev_node, current_node, **g.edges[prev_node, current_node]
                )
            else:
                # Reset
                if start_node is None:
                    current_node = random.choice(possible_nodes)
                else:
                    current_node = start_node

            if num_steps is not None and counter >= num_steps:
                break
            if num_nodes is not None and len(out_g) >= num_nodes:
                break
            if num_edges is not None and out_g.size() >= num_edges:
                break

        return NetworkXGraph(out_g)


if has_networkx and has_community:
    import community as community_louvain
    from .types import NetworkXGraph
    from ..python.types import PythonNodeMapType

    @concrete_algorithm("clustering.louvain_community")
    def nx_louvain_community(graph: NetworkXGraph) -> Tuple[PythonNodeMapType, float]:
        index_to_label = community_louvain.best_partition(graph.value)
        modularity_score = community_louvain.modularity(index_to_label, graph.value)
        return (
            index_to_label,
            modularity_score,
        )


if has_networkx and has_pandas:
    from ..pandas.types import PandasEdgeSet, PandasEdgeMap
    from ..python.types import PythonNodeMapType, PythonNodeSetType

    @concrete_algorithm("util.graph.build")
    def nx_graph_build_from_pandas(
        edges: mg.Union[PandasEdgeSet, PandasEdgeMap],
        nodes: mg.Optional[mg.Union[PythonNodeSetType, PythonNodeMapType]],
    ) -> NetworkXGraph:
        g = nx.DiGraph() if edges.is_directed else nx.Graph()
        if nodes is not None:
            if PythonNodeMapType.is_typeclass_of(nodes):
                g.add_nodes_from((n, {"weight": v}) for n, v in nodes.items())
            else:
                g.add_nodes_from(nodes)
        if type(edges) is PandasEdgeMap:
            df = edges.value[[edges.src_label, edges.dst_label, edges.weight_label]]
            g.add_weighted_edges_from(df.itertuples(index=False, name="WeightedEdge"))
        else:
            df = edges.value[[edges.src_label, edges.dst_label]]
            g.add_edges_from(df.itertuples(index=False, name="Edge"))
        return NetworkXGraph(g)
