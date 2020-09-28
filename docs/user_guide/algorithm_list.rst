.. _algorithm_list:

List of Core Algorithms
=======================

This document lists the core abstract algorithms in Metagraph.

Clustering
----------

Graphs often have natural structure which can be discovered, allowing them to be clustered into different groups or partitions.

.. py:function:: clustering.connected_components(graph: Graph(is_directed=False)) -> NodeMap

    The connected components algorithm groups nodes of an **undirected** graph into subgraphs where all subgraph nodes
    are reachable within a component.

    :rtype: a dense NodeMap where each node is assigned an integer indicating the component.


.. py:function:: clustering.strongly_connected_components(graph: Graph(is_directed=True)) -> NodeMap

    Groups nodes of a directed graph into subgraphs where all subgraph nodes are reachable by each other along directed edges.

    :rtype: a dense NodeMap where each node is assigned an integer indicating the component.


.. py:function:: clustering.label_propagation_community(graph: Graph(is_directed=False)) -> NodeMap

    This algorithm discovers communities using `label propagation <https://en.wikipedia.org/wiki/Label_propagation_algorithm>`_.

    :rtype: a dense NodeMap where each node is assigned an integer indicating the community.


.. py:function:: clustering.louvain_community_step(graph: Graph(is_directed=False, edge_type="map", edge_dtype={"int", "float"})) -> Tuple[NodeMap, float]

    This algorithm performs one step of the `Louvain algorithm <https://en.wikipedia.org/wiki/Louvain_modularity>`_,
    which discovers communities by maximizing modularity.

    :rtype:
      - a dense NodeMap where each node is assigned an integer indicating the community
      - the modularity score


.. py:function:: cluster.triangle_count(graph: Graph(is_directed=False)) -> int

    This algorithms returns the total number of triangles in the graph.


Traversal
---------

Traversing through the nodes of a graph is extremely common and important in the area of search and understanding distance between nodes.


.. py:function:: traversal.bellman_ford(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), source_node: NodeID) -> Tuple[NodeMap, NodeMap]

    This algorithm calculates `single-source shortest path (SSSP) <https://en.wikipedia.org/wiki/Shortest_path_problem>`_.
    It is slower than `Dijkstra’s algorithm <https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm>`_, but can handle
    negative weights and is parallelizable.

    :rtype: (parents, distance)



.. py:function:: traversal.all_pairs_shortest_paths(graph: Graph(edge_type="map", edge_dtype={"int", "float"})) -> Tuple[Graph, Graph]

    This algorithm calculates the shortest paths between all node pairs. Choices for which algorithm to be used are
    backend implementation dependent.

    :rtype: (parents, distance)


.. py:function:: traversal.bfs_iter(graph: Graph, source_node: NodeID, depth_limit: int = 1) -> Vector

    Breadth-first search algorithm.

    :rtype: Node IDs in search order


.. py:function:: traversal.bfs_tree(graph: Graph, source_node: NodeID, depth_limit: int = 1) -> Tuple[NodeMap, NodeMap]

    Breadth-first search algorithm.

    :rtype: (depth, parents)


.. py:function:: traversal.dijkstra(graph: Graph(edge_type="map", edge_dtype={"int", "float"}, edge_has_negative_weights=False), source_node: NodeID) -> Tuple[NodeMap, NodeMap]

    Calculates `single-source shortest path (SSSP) <https://en.wikipedia.org/wiki/Shortest_path_problem>`_ via
    `Dijkstra's algorithm <https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm>`_.

    :rtype: (parents, distance)


.. py:function:: traversal.minimum_spanning_tree(graph: Graph(is_directed=False, edge_type="map", edge_dtype={"int", "float"})) -> Graph

    Minimum spanning tree (or forest in the case of multiple connected components in the graph).

    :rtype: Graph containing only the relevant edges from the original graph


Centrality
----------

Many algorithms assign a ranking or value to each vertex/node in the graph based on different properties. This is usually done to find the most important nodes for that metric.


.. py:function:: centrality.betweenness(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), nodes: Optional[NodeSet] = None, normalize: bool = False) -> NodeMap

    This algorithm calculates centrality based on the number of shortest paths passing through a node.

    If ``nodes`` are provided, only computes an approximation of betweenness centrality based on those nodes.


.. py:function:: centrality.katz(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), attenuation_factor: float = 0.01, immediate_neighbor_weight: float = 1.0, maxiter: int = 50, tolerance: float = 1e-05) -> NodeMap

    This algorithm calculates centrality based on total number of walks (as opposed to only considering shortest paths) passing through a node.


.. py:function:: centrality.pagerank(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), damping: float = 0.85, maxiter: int = 50, tolerance: float = 1e-05) -> NodeMap

    This algorithm determines the importance of a given node in the network based on links between important nodes.


Subgraph
--------

Graphs are often too large to handle, so a portion of the graph is extracted. Often this subgraph must satisfy certain properties or have properties similar to the original graph for the subsequent analysis to give good results.


.. py:function:: subgraph.extract_subgraph(graph: Graph, nodes: NodeSet) -> Graph

    Given a set of nodes, this algorithm extracts the subgraph containing those nodes and any edges between those nodes.


.. py:function:: subgraph.k_core(graph: Graph(is_directed=False), k: int) -> Graph

    This algorithm finds a maximal subgraph that contains nodes of at least degree *k*.


Bipartite
---------

Bipartite Graphs contain two unique sets of nodes. Edges can exist between nodes from different groups, but not between nodes of the same group.

.. py:function:: bipartite.graph_projection(bgraph: BipartiteGraph, nodes_retained: int = 0) -> Graph

    Given a bipartite graph, project a graph for one of the two node groups (group 0 or 1).


Flow
----

Algorithms pertaining to the flow capacity of edges.

.. py:function:: flow.max_flow(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), source_node: NodeID, target_node: NodeID) -> Tuple[float, Graph]

    Compute the maximum flow possible from source_node to target_node

    :rtype: (max_flow_rate, compute_flow_graph)


Utility
-------

These algorithms are small utility functions which perform common operations needed in graph analysis.

.. py:function:: util.nodeset.choose_random(x: NodeSet, k: int) -> NodeSet

    Given a set of nodes, choose k random nodes (no duplicates).

.. py:function:: util.nodeset.from_vector(x: Vector) -> NodeSet

    Convert the values in a Vector into a NodeSet

.. py:function:: util.nodemap.sort(x: NodeMap, ascending: bool = True, limit: Optional[int] = None) -> Vector

    Sorts nodes by value, returning a Vector of NodeIDs.

.. py:function:: util.nodemap.select(x: NodeMap, nodes: NodeSet) -> NodeMap

    Selects certain nodes to keep from a NodeMap.

.. py:function:: util.nodemap.filter(x: NodeMap, func: Callable[[Any], bool]) -> NodeSet

    Filters a NodeMap based on values passed through the filter function. Returns a set of nodes where the function returned True.

.. py:function:: util.nodemap.apply(x: NodeMap, func: Callable[[Any], Any]) -> NodeMap

    Applies a unary function to every node, mapping the values to different values.

.. py:function:: util.nodemap.reduce(x: NodeMap, func: Callable[[Any, Any], Any]) -> Any

    Performs a reduction across all nodes, collapsing the values into a single result.

.. py:function:: util.edgemap.from_edgeset(edgeset: EdgeSet, default_value: Any) -> EdgeMap

    Converts and EdgeSet into an EdgeMap by giving each edge a default value.

.. py:function:: util.graph.aggregate_edges(graph: Graph(edge_type="map"), func: Callable[[Any, Any], Any]), initial_value: Any, in_edges: bool = False, out_edges: bool = True) -> NodeMap

    Aggregates the edge weights around a node, returning a single value per node.

    If ``in_edges`` and ``out_edges`` are False, each node will contain the initial value.
    For undirected graphs, setting ``in_edges`` or ``out_edges`` or both will give identical results. Edges will only be counted once per node.
    For directed graphs, ``in_edges`` and ``out_edges`` affect the result. Setting both will still only give a single value per node, combining all outbound and inbound edge weights.

.. py:function:: util.graph.filter_edges(graph: Graph(edge_type="map"), func: Callable[[Any], bool]) -> Graph

    Removes edges if filter function returns True.
    All nodes remain, even if they becomes orphan nodes in the graph.

.. py:function:: util.graph.assign_uniform_weight(graph: Graph, weight: Any = 1) -> Graph(edge_type="map")

    Update all edge weights (or if none exist, assign them) to a uniform value of ``weight``.

.. py:function:: util.graph.build(edges: Union[EdgeSet, EdgeMap], nodes: Optional[Union[NodeSet, NodeMap]] = None) -> Graph

    Given edges and possibly nodes, build a Graph.

    If ``nodes`` are not provided, assume the only nodes are those found in the EdgeSet/Map.

.. py:function:: util.graph.collapse_by_label(graph: Graph(is_directed=False), labels: NodeMap, aggregator: Callable[[Any, Any], Any]) -> Graph

    Collapse a Graph into a smaller Graph by combining clusters of nodes into a single node.
    ``labels`` indicates the node groupings. ``aggregator`` indicates how to combine edge weights.
