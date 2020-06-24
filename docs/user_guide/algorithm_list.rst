List of Core Algorithms
=======================

This document will go over the core abstract algorithms that come with metagraph.

Clustering
----------

Graphs often have natural structure which can be discovered, allowing them to be clustered into different groups or partitions.


Connected Components
~~~~~~~~~~~~~~~~~~~~

 .. code-block:: python
		 
		 def connected_components(graph: EdgeSet(is_directed=False)) -> NodeMap

The connected components algorithm groups nodes of an **undirected** graph into subgraphs where all subgraph nodes which are reachable by each other. Since the graph is undirected, the subgraphs are both weakly and strongly connected.

Strongly Connected Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. code-block:: python
		 
		 def strongly_connected_components(graph: EdgeSet(is_directed=True)) -> NodeMap:

Groups nodes of a directed graph into subgraphs where all subgraph nodes which are reachable by each other.

Label Propagation
~~~~~~~~~~~~~~~~~

.. code-block:: python

		def label_propagation_community(graph: EdgeMap(is_directed=False)) -> NodeMap:

This algorithm discovers communities using `label propagation <https://en.wikipedia.org/wiki/Label_propagation_algorithm>`_.

Louvain Community Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

		def louvain_community(graph: EdgeMap(is_directed=False)) -> Tuple[NodeMap, float]:

This algorithm performs one step of the `Louvain algorithm <https://en.wikipedia.org/wiki/Louvain_modularity>`_, which discovers communities by maximizing modularity.

Triangle Count
~~~~~~~~~~~~~~

.. code-block:: python

		def triangle_count(graph: EdgeSet(is_directed=False)) -> int:

This algorithms returns the total triangle count in the graph.

Traversal
---------

Traversing through the nodes of a graph is extremely common and important in the area of search and understanding distance between nodes.

Bellman-Ford Single Source Shortest Paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

		def bellman_ford(graph: EdgeMap, source_node: NodeID) -> Tuple[NodeMap, NodeMap]:

This algorithm calculates `single-source shortest path (SSSP) <https://en.wikipedia.org/wiki/Shortest_path_problem>`_. It is slower than `Dijkstra’s algorithm <https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm>`_, but can handle negative weights and is parallelizable.

All Shortest Paths
~~~~~~~~~~~~~~~~~~

.. code-block:: python

		def all_shortest_paths(graph: EdgeMap) -> Tuple[EdgeMap, EdgeMap]:

This algorithm calculates the shortest paths between all node pairs. Choices for which algorithm to be used are backend implementation dependent.

Breadth First Search
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

		def breadth_first_search(graph: EdgeSet, source_node: NodeID) -> Vector:

This is the breadth first search algorithm.

Dijkstra Single Source Shortest Paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

		def dijkstra(graph: EdgeMap(has_negative_weights=False), source_node: NodeID, max_path_length: float) -> Tuple[NodeMap, NodeMap]:

Calculates `single-source shortest path (SSSP) <https://en.wikipedia.org/wiki/Shortest_path_problem>` via `Dijkstra's algorithm <https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm>`_.

Vertex Ranking
--------------

Many algorithms assign a ranking or value to each vertex/node in the graph based on different properties. This is usually done to find the most important nodes for that metric.

Betweenness Centrality
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

		def betweenness_centrality(graph: EdgeMap(dtype={"int", "float"}), k: int, enable_normalization: bool, include_endpoints: bool) -> NodeMap:

This algorithm calculates centrality based on the number of shortest paths passing through a node.

Katz Centrality
~~~~~~~~~~~~~~~

.. code-block:: python

		def katz_centrality(graph: EdgeMap(dtype={"int", "float"}), attenuation_factor: float = 0.01, immediate_neighbor_weight: float = 1.0, maxiter: int = 50, tolerance: float = 1e-05) -> NodeMap:

This algorithm calculates centrality based on total number of walks (as opposed to only considering shortest paths) passing through a node.

Page Rank
~~~~~~~~~

.. code-block:: python

		def pagerank(graph: EdgeMap(dtype={"int", "float"}), damping: float = 0.85, maxiter: int = 50, tolerance: float = 1e-05) -> NodeMap:

This algorithm determiens the importance of a given node in the network based on links between important nodes.

Subgraph
--------

Graphs are often too large to handle, so a portion of the graph is extracted. Often this subgraph must satisfy certain properties or have properties similar to the original graph for the subsequent analysis to give good results.

Subgraph Extraction (Weighted) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

		def extract_edgemap(graph: EdgeMap, nodes: NodeSet) -> EdgeMap:

Given a set of nodes, this algorithm extracts the subgraph of a weighted graph containing those nodes and any edges between those nodes.

Subgraph Extraction (Unweighted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

		def extract_edgeset(graph: EdgeSet, nodes: NodeSet) -> EdgeSet:

Given a set of nodes, this algorithm extracts the subgraph of an unweighted graph containing those nodes and any edges between those nodes.

K-Core (Weighted)
~~~~~~~~~~~~~~~~~

.. code-block:: python

		def k_core(graph: EdgeMap, k: int) -> EdgeMap:

This algorith finds a maximal subgraph of a given weighted graph that contains nodes of at least degree *k*.


K-Core (Unweighted)
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

		def k_core_unweighted(graph: EdgeSet, k: int) -> EdgeSet:

This algorithm finds a maximal subgraph of a given unweighted graph that contains nodes of at least degree *k*.

