.. _algorithm_list:

List of Core Algorithms
=======================

This document lists the core abstract algorithms in Metagraph.

Clustering
----------

Graphs often have natural structure which can be discovered, allowing them to be clustered into different groups or partitions.

.. py:function:: clustering.connected_components(graph: Graph(is_directed=False)) -> NodeMap

    The connected components algorithm groups nodes of an undirected graph into subgraphs where all subgraph nodes
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


.. py:function:: clustering.triangle_count(graph: Graph(is_directed=False)) -> int

    This algorithm returns the total number of triangles in the graph.

    
.. py:function:: clustering.global_clustering_coefficient(graph: Graph(is_directed=False)) -> float

    This algorithm returns the global clustering coefficient. The global clustering coefficient is the number of closed
    triplets over the total number of triplets in a graph. A triplet in a graph is a subgraph of 3 nodes where at least
    2 edges are present. An open triplet has exactly 2 edges. A closed triplet has exactly 3 edges. A deeped explanation
    can be found `here <https://en.wikipedia.org/wiki/Clustering_coefficient#Global_clustering_coefficient>`_.


.. py:function:: clustering.coloring.greedy(graph: Graph(is_directed=False)) -> Tuple[NodeMap, int]

    Attempts to find the minimum number of colors required to label the graph such that no connected nodes have the
    same color. Color is represented as a value from 0..n.

    :rtype: (color for each node, number of unique colors)


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


.. py:function:: traversal.bfs_iter(graph: Graph, source_node: NodeID, depth_limit: int = -1) -> Vector

    Breadth-first search algorithm.

    :rtype: Node IDs in breadth-first search order


.. py:function:: traversal.bfs_tree(graph: Graph, source_node: NodeID, depth_limit: int = -1) -> Tuple[NodeMap, NodeMap]

    Breadth-first search algorithm. The return result ``parents`` will have the parent of ``source_node`` be ``source_node``.

    :rtype: (depth, parents)


.. py:function:: traversal.dfs_iter(graph: Graph, source_node: NodeID) -> Vector

    Depth-first search algorithm.

    :rtype: Node IDs in depth-first search order


.. py:function:: traversal.dfs_tree(graph: Graph, source_node: NodeID) -> NodeMap

    Depth-first search algorithm. The return result ``parents`` will have the parent of ``source_node`` be ``source_node``.

    :rtype: parents


.. py:function:: traversal.dijkstra(graph: Graph(edge_type="map", edge_dtype={"int", "float"}, edge_has_negative_weights=False), source_node: NodeID) -> Tuple[NodeMap, NodeMap]

    Calculates `single-source shortest path (SSSP) <https://en.wikipedia.org/wiki/Shortest_path_problem>`_ via
    `Dijkstra's algorithm <https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm>`_.

    :rtype: (parents, distance)


.. py:function:: traversal.minimum_spanning_tree(graph: Graph(is_directed=False, edge_type="map", edge_dtype={"int", "float"})) -> Graph

    Minimum spanning tree (or forest in the case of multiple connected components in the graph).

    :rtype: Graph containing only the relevant edges from the original graph

.. py:function:: traversal.astar_search(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), source_node: NodeID, target_node: NodeID, heuristic_func: Callable[[NodeID], float]) -> Vector

    Finds the (possibly non-unique) shortest path via the `A* algorithm <https://en.wikipedia.org/wiki/A*_search_algorithm>`_.
    ``heuristic_func`` is a unary function that takes a node id and returns an estimated distance to ``target_node``.

    :rtype: Vector of node ids specifying the path from ``source_node`` to ``target_node``


Centrality
----------

Many algorithms assign a ranking or value to each vertex/node in the graph based on different properties. This is
usually done to find the most important nodes for that metric.


.. py:function:: centrality.betweenness(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), nodes: Optional[NodeSet] = None, normalize: bool = False) -> NodeMap

    This algorithm calculates centrality based on the number of shortest paths passing through a node.

    If ``nodes`` are provided, only computes an approximation of betweenness centrality based on those nodes.


.. py:function:: centrality.katz(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), attenuation_factor: float = 0.01, immediate_neighbor_weight: float = 1.0, maxiter: int = 50, tolerance: float = 1e-05) -> NodeMap

    This algorithm calculates centrality based on total number of walks (as opposed to only considering shortest paths)
    passing through a node.


.. py:function:: centrality.pagerank(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), damping: float = 0.85, maxiter: int = 50, tolerance: float = 1e-05) -> NodeMap

    This algorithm determines the importance of a given node in the network based on links between important nodes.


.. py:function:: centrality.closeness(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), nodes: Optional[NodeSet] = None) -> NodeMap

    Calculates the closeness centrality metric, which estimates the average distance from a node to all other nodes.
    A high closeness score indicates a small average distance to other nodes.

.. py:function:: centrality.eigenvector(graph: Graph(edge_type="map", edge_dtype={"int", "float"})) -> NodeMap

    Calculates the eigenvector centrality, which estimates the importance of a node in the graph.

.. py:function:: centrality.hits(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), max_iter: int = 100, tol: float = 1e-05, normalize: bool = True) -> Tuple[NodeMap, NodeMap]

    Hyperlink-Induced Topic Search (HITS) centrality ranks nodes based on incoming and outgoing edges.

    :rtype: (hubs, authority)

.. py:function:: centrality.degree(graph: Graph, in_edges: bool = False, out_edges: bool = True) -> NodeMap

    Calculates the degree centrality for each node. The degree centrality for a node is its degree over the number of nodes minus 1.

    If ``in_edges`` and ``out_edges`` are both false, the degree centrality for all nodes is 0.
    If the graph is undirected, setting ``in_edges`` or ``out_edges`` or both to true will give identical results
    (edges will only be counted once per node).
    If the graph is directed, ``in_edges`` and ``out_edges`` dictate which edges are considered for each node. 


Subgraph
--------

Graphs are often too large to handle, so a portion of the graph is extracted. Often this subgraph must satisfy certain
properties or have properties similar to the original graph for the subsequent analysis to give good results.


.. py:function:: subgraph.extract_subgraph(graph: Graph, nodes: NodeSet) -> Graph

    Given a set of nodes, this algorithm extracts the subgraph containing those nodes and any edges between those nodes.


.. py:function:: subgraph.k_core(graph: Graph(is_directed=False), k: int) -> Graph

    This algorithm finds a maximal subgraph that contains nodes of at least degree ``k``.


.. py:function:: subgraph.k_truss(graph: Graph(is_directed=False), k: int) -> Graph

    Finds the maximal subgraph whose edges are supported by ``k`` - 2 other edges forming triangles.


.. py:function:: subgraph.maximal_independent_set(graph: Graph) -> NodeSet

    Finds a maximal set of independent nodes, meaning the nodes in the set share no edges with each other
    and no additional nodes in the graph can be added which satisfy this criteria.


.. py:function:: subgraph.subisomorphic(graph: Graph, subgraph: Graph) -> bool

    Indicates whether ``subgraph`` is an isomorphic subcomponent of ``graph``.


.. py:function:: subgraph.sample.node_sampling(graph: Graph, p: float = 0.20) -> Graph

    Returns a subgraph created by randomly sampling nodes and including edges which exist between sampled
    nodes in the original graph.


.. py:function:: subgraph.sample.edge_sampling(graph: Graph, p: float = 0.20) -> Graph

    Returns a subgraph created by randomly sampling edges and including both node endpoints.


.. py:function:: subgraph.sample.ties(graph: Graph, p: float = 0.20) -> Graph

    Totally Induced Edge Sampling extends edge sampling by also including any edges between the nodes
    which exist in the original graph. See the `paper <https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2743&context=cstech>`__
    for more details.


.. py:function:: subgraph.sample.random_walk(graph: Graph, num_steps: Optional[int] = None, num_nodes: Optional[int] = None, num_edges: Optional[int] = None, jump_probability: int = 0.15, start_node: Optional[NodeID] = None) -> Graph

    Samples the graph using a random walk. For each step, there is a ``jump_probability`` to reset the walk.
    When resetting the walk, if the ``start_node`` is specified, it always returns to this node. Otherwise a random
    node is chosen for each resetting. The sampling stops when any of ``num_steps``, ``num_nodes``, or ``num_edges`` is
    reached.



Bipartite
---------

Bipartite Graphs contain two unique sets of nodes. Edges can exist between nodes from different groups, but not between
nodes of the same group.

.. py:function:: bipartite.graph_projection(bgraph: BipartiteGraph, nodes_retained: int = 0) -> Graph

    Given a bipartite graph, project a graph for one of the two node groups (group 0 or 1).


Flow
----

Algorithms pertaining to the flow capacity of edges.

.. py:function:: flow.max_flow(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), source_node: NodeID, target_node: NodeID) -> Tuple[float, Graph]

    Compute the maximum flow possible from ``source_node`` to ``target_node``.

    :rtype: (max flow rate, computed flow graph)


.. py:function:: flow.min_cut(graph: Graph(edge_type="map", edge_dtype={"int", "float"}), source_node: NodeID, target_node: NodeID) -> Tuple[float, Graph]

    Compute the minimum cut to separate source from target node. This is the list of edges which disconnect the graph
    along edges with sum to the minimum weight.
    Performing this computation yields the maximum flow.

    :rtype: (max flow rate, graph containing cut edges)


Utility
-------

These algorithms are small utility functions which perform common operations needed in graph analysis.

.. py:function:: util.nodeset.choose_random(x: NodeSet, k: int) -> NodeSet

    Given a set of nodes, choose ``k`` random nodes (no duplicates).

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

.. py:function:: util.graph.degree(graph: Graph, in_edges: bool = False, out_edges: bool = True) -> NodeMap

    Computes the degree of each node. ``in_edges`` and ``out_edges`` can be used to control which degree is computed.

.. py:function:: util.graph.aggregate_edges(graph: Graph(edge_type="map"), func: Callable[[Any, Any], Any]), initial_value: Any, in_edges: bool = False, out_edges: bool = True) -> NodeMap

    Aggregates the edge weights around a node, returning a single value per node.

    If ``in_edges`` and ``out_edges`` are False, each node will contain the initial value.
    For undirected graphs, setting ``in_edges`` or ``out_edges`` or both to true will give identical results
    (edges will only be counted once per node).
    For directed graphs, ``in_edges`` and ``out_edges`` affect the result. Setting both will still only give a single
    value per node, combining all outbound and inbound edge weights.

.. py:function:: util.graph.filter_edges(graph: Graph(edge_type="map"), func: Callable[[Any], bool]) -> Graph

    Removes edges if filter function returns True.
    All nodes remain, even if they becomes isolate nodes in the graph.

.. py:function:: util.graph.assign_uniform_weight(graph: Graph, weight: Any = 1) -> Graph(edge_type="map")

    Update all edge weights (or if none exist, assign them) to a uniform value of ``weight``.

.. py:function:: util.graph.build(edges: Union[EdgeSet, EdgeMap], nodes: Optional[Union[NodeSet, NodeMap]] = None) -> Graph

    Given edges and possibly nodes, build a Graph.

    If ``nodes`` are not provided, assume the only nodes are those found in the EdgeSet/Map.

.. py:function:: util.graph.collapse_by_label(graph: Graph(is_directed=False), labels: NodeMap, aggregator: Callable[[Any, Any], Any]) -> Graph

    Collapse a Graph into a smaller Graph by combining clusters of nodes into a single node.
    ``labels`` indicates the node groupings. ``aggregator`` indicates how to combine edge weights.

.. py:function:: util.graph.isomorphic(g1: Graph, g2: Graph) -> bool

    Indicates whether ``g1`` and ``g2`` are isomorphic.


Embedding
---------

Embeddings convert graph nodes or whole graphs into a dense vector representations.

.. py:function:: embedding.apply.nodes(matrix: Matrix, node2row: NodeMap, nodes: Vector) -> Matrix

    Returns a dense matrix given an embedding, node-to-row mapping, and a vector of NodeIDs.

.. py:function:: embedding.apply.graph_sage(embedding: GraphSageNodeEmbedding, graph: Graph, node_features: Matrix, node2row: NodeMap) -> Matrix

    Returns a dense matrix from a GraphSage embedding.

.. py:function:: embedding.train.node2vec(graph: Graph, p: float, q: float, walks_per_node: int, walk_length: int, embedding_size: int, epochs: int, learning_rate: float) -> Tuple[Matrix, NodeMap]

    Computes the `node2vec <https://snap.stanford.edu/node2vec/>`__ embedding.

.. py:function:: embedding.train.graph2vec(graphs: mg.List[Graph(edge_type="set", is_directed=False)], subgraph_degree: int, embedding_size: int, epochs: int, learning_rate: float) -> Matrix

    Computes the `graph2vec <https://arxiv.org/abs/1707.05005>`__ embedding.

.. py:function:: embedding.train.graphwave(graph: Graph(edge_type="set", is_directed=False), scales: Vector, sample_point_count: int, sample_point_max: float, chebyshev_degree: int) -> Tuple[Matrix, NodeMap]

    Computes the `graphwave <http://snap.stanford.edu/graphwave/>`__ embedding.

.. py:function:: embedding.train.hope.katz(graph: Graph(edge_type="map", is_directed=True), embedding_size: int, beta: float) -> Tuple[Matrix, NodeMap]

    Computes the `High-Order Proximity preserved Embedding <https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf>`__ (HOPE).

.. py:function:: embedding.train.graph_sage.mean(graph: Graph(edge_type="map", is_directed=True), node_features: Matrix, node2row: NodeMap, walk_length: int, walks_per_node: int, layer_sizes: Vector, samples_per_layer: Vector, epochs: int, learning_rate: float, batch_size: int) -> GraphSageNodeEmbedding

    Computes the `GraphSAGE <http://snap.stanford.edu/graphsage/>`__ embedding.

.. py:function:: embedding.train.line(graph: Graph, walks_per_node: int, negative_sample_count: int, embedding_size: int, epochs: int, learning_rate: float, batch_size: int) -> Tuple[Matrix, NodeMap]

    Computes the `Large-scale Information Network Embedding <https://arxiv.org/abs/1503.03578>`__ (LINE).
