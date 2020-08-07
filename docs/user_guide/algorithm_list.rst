.. _algorithm_list:

List of Core Algorithms
=======================

This document lists the core abstract algorithms in Metagraph.

Clustering
----------

Graphs often have natural structure which can be discovered, allowing them to be clustered into different groups or partitions.

.. py:function:: connected_components(graph: Graph(is_directed=False)) -> NodeMap

    The connected components algorithm groups nodes of an **undirected** graph into subgraphs where all subgraph nodes
    are reachable within a component.

    :rtype: a dense NodeMap where each node is assigned an integer indicating the component.


.. py:function:: strongly_connected_components(graph: Graph(is_directed=True)) -> NodeMap

    Groups nodes of a directed graph into subgraphs where all subgraph nodes are reachable by each other along directed edges.

    :rtype: a dense NodeMap where each node is assigned an integer indicating the component.


.. py:function:: label_propagation_community(graph: Graph(is_directed=False)) -> NodeMap

    This algorithm discovers communities using `label propagation <https://en.wikipedia.org/wiki/Label_propagation_algorithm>`_.

    :rtype: a dense NodeMap where each node is assigned an integer indicating the community.


.. py:function:: louvain_community(graph: Graph(is_directed=False)) -> Tuple[NodeMap, float]

    This algorithm performs one step of the `Louvain algorithm <https://en.wikipedia.org/wiki/Louvain_modularity>`_,
    which discovers communities by maximizing modularity.

    :rtype:
      - a dense NodeMap where each node is assigned an integer indicating the community
      - the modularity score


.. py:function:: triangle_count(graph: Graph(is_directed=False)) -> int

    This algorithms returns the total number of triangles in the graph.


Traversal
---------

Traversing through the nodes of a graph is extremely common and important in the area of search and understanding distance between nodes.


.. py:function:: bellman_ford(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), source_node: NodeID) -> Tuple[NodeMap, NodeMap]

    This algorithm calculates `single-source shortest path (SSSP) <https://en.wikipedia.org/wiki/Shortest_path_problem>`_.
    It is slower than `Dijkstra’s algorithm <https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm>`_, but can handle
    negative weights and is parallelizable.

    :rtype: (parents, distance)



.. py:function:: all_pairs_shortest_paths(graph: Graph(edge_type="map", edge_dtype={"int", "float"})) -> Tuple[EdgeMap, EdgeMap]

    This algorithm calculates the shortest paths between all node pairs. Choices for which algorithm to be used are
    backend implementation dependent.

    :rtype: (parents, distance)


.. py:function:: bfs_iter(graph: Graph, source_node: NodeID, depth_limit: int = 1) -> Vector

    Breadth-first search algorithm.

    :rtype: Node IDs in search order


.. py:function:: bfs_tree(graph: Graph, source_node: NodeID, depth_limit: int = 1) -> Tuple[NodeMap, NodeMap]

    Breadth-first search algorithm.

    :rtype: (depth, parents)


.. py:function:: dijkstra(graph: Graph(edge_type="map", edge_dtype={"int", "float"}, edge_has_negative_weights=False), source_node: NodeID) -> Tuple[NodeMap, NodeMap]

    Calculates `single-source shortest path (SSSP) <https://en.wikipedia.org/wiki/Shortest_path_problem>`_ via
    `Dijkstra's algorithm <https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm>`_.

    :rtype: (parents, distance)


Centrality
----------

Many algorithms assign a ranking or value to each vertex/node in the graph based on different properties. This is usually done to find the most important nodes for that metric.


.. py:function:: betweenness(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), nodes: Optional[NodeSet] = None, normalize: bool = False) -> NodeMap

    This algorithm calculates centrality based on the number of shortest paths passing through a node.

    If ``nodes`` are provided, only computes an approximation of betweenness centrality based on those nodes.


.. py:function:: katz(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), attenuation_factor: float = 0.01, immediate_neighbor_weight: float = 1.0, maxiter: int = 50, tolerance: float = 1e-05) -> NodeMap

    This algorithm calculates centrality based on total number of walks (as opposed to only considering shortest paths) passing through a node.


.. py:function:: pagerank(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), damping: float = 0.85, maxiter: int = 50, tolerance: float = 1e-05) -> NodeMap

    This algorithm determines the importance of a given node in the network based on links between important nodes.


Subgraph
--------

Graphs are often too large to handle, so a portion of the graph is extracted. Often this subgraph must satisfy certain properties or have properties similar to the original graph for the subsequent analysis to give good results.


.. py:function:: extract_subgraph(graph: Graph, nodes: NodeSet) -> Graph

    Given a set of nodes, this algorithm extracts the subgraph containing those nodes and any edges between those nodes.


.. py:function:: k_core(graph: Graph(is_directed=False), k: int) -> Graph

    This algorithm finds a maximal subgraph that contains nodes of at least degree *k*.


Bipartite
---------

Bipartite Graphs contain two unique sets of nodes. Edges can exist between nodes from different groups, but not between nodes of the same group.

.. py:function:: graph_projection(bgraph: BipartiteGraph, nodes_retained: int = 0) -> Graph

    Given a bipartite graph, project a graph for one of the two node groups (group 0 or 1).


Utility
-------

These algorithms are small utility functions which perform common operations needed in graph analysis.

.. py:function:: nodeset_choose_random(x: NodeSet, k: int) -> NodeSet

    Given a set of nodes, choose k random nodes (no duplicates).

.. py:function:: nodemap_sort(x: NodeMap, ascending: bool = True, limit: Optional[int] = None) -> Vector

    Sorts nodes by value, returning a Vector of NodeIDs.

.. py:function:: nodemap_select(x: NodeMap, nodes: NodeSet) -> NodeMap

    Selects certain nodes to keep from a NodeMap.

.. py:function:: nodemap_filter(x: NodeMap, func: Callable[[Any], bool]) -> NodeSet

    Filters a NodeMap based on values passed through the filter function. Returns a set of nodes where the function returned True.

.. py:function:: nodemap_apply(x: NodeMap, func: Callable[[Any], Any]) -> NodeMap

    Applies a unary function to every node, mapping the values to different values.

.. py:function:: nodemap_reduce(x: NodeMap, func: Callable[[Any, Any], Any]) -> Any

    Performs a reduction across all nodes, collapsing the values into a single result.

.. py:function:: graph_aggregate_edges(graph: Graph(edge_type="map"), func: Callable[[Any, Any], Any]), inintial_value: Any, in_edges: bool = False, out_edges: bool = True) -> NodeMap

    Aggregates the edge weights around a node, returning a single value per node.

    If in_edges and out_edges are False, each node will contain the initial value.
    For undirected graphs, setting in_edges or out_edges or both will give identical results. Edges will only be counted once per node.
    For directed graphs, in_edges and out_edges affect the result. Setting both will still only give a single value per node, combining all outbound and inbound edge weights.

.. py:function:: graph_filter_edges(graph: Graph(edge_type="map"), func: Callable[[Any], bool]) -> Graph

    Removes edges if filter function returns True.
    All nodes remain, even if they becomes orphan nodes in the graph.

.. py:function:: graph_assign_uniform_weight(graph: Graph, weight: Any = 1) -> Graph(edge_type="map")

    Update all edge weights (or if none exist, assign them) to a uniform value of ``weight``.

.. py:function:: graph_build(edges: Union[EdgeSet, EdgeMap], nodes: Optional[Union[NodeSet, NodeMap]] = None) -> Graph

    Given edges and possibly nodes, build a Graph.

    If nodes are not provided, assume the only nodes are those found in the EdgeSet/Map.
