from metagraph import abstract_algorithm
from metagraph.types import Graph, Vector, NodeMap, NodeID
from typing import Tuple, Callable


@abstract_algorithm("traversal.bellman_ford")
def bellman_ford(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}), source_node: NodeID
) -> Tuple[NodeMap, NodeMap]:
    """Output is (Parents, Distance)"""
    pass  # pragma: no cover


@abstract_algorithm("traversal.all_pairs_shortest_paths")
def all_pairs_shortest_paths(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"})
) -> Tuple[Graph, Graph]:
    """Output is (Parents, Distance)"""
    pass  # pragma: no cover


@abstract_algorithm("traversal.bfs_iter")
def breadth_first_search_iterator(
    graph: Graph, source_node: NodeID, depth_limit: int = -1
) -> Vector:
    """Output is NodeID"""
    pass  # pragma: no cover


@abstract_algorithm("traversal.bfs_tree")
def breadth_first_search_tree(
    graph: Graph, source_node: NodeID, depth_limit: int = -1
) -> Tuple[NodeMap, NodeMap]:
    """
    Output is (NodeID->Depth, NodeID->Parent)
    NodeID->Parent will include source_node, whose parent will be itself.
    """
    pass  # pragma: no cover


@abstract_algorithm("traversal.dfs_iter")
def depth_first_search_iterator(graph: Graph, source_node: NodeID) -> Vector:
    """Output is NodeID"""
    pass  # pragma: no cover


@abstract_algorithm("traversal.dfs_tree")
def depth_first_search_tree(graph: Graph, source_node: NodeID) -> NodeMap:
    """
    Returns NodeID->Parent, which will include source_node (whose parent will be itself).
    """
    pass  # pragma: no cover


@abstract_algorithm("traversal.dijkstra")
def dijkstra(
    graph: Graph(
        edge_has_negative_weights=False, edge_type="map", edge_dtype={"int", "float"}
    ),
    source_node: NodeID,
) -> Tuple[NodeMap, NodeMap]:
    """Output is (Parents, Distance)"""
    pass  # pragma: no cover


@abstract_algorithm("traversal.minimum_spanning_tree")
def minimum_spanning_tree(
    graph: Graph(is_directed=False, edge_type="map", edge_dtype={"int", "float"})
) -> Graph(is_directed=False, edge_type="map", edge_dtype={"int", "float"}):
    """If the graph is disconnected, a minimum spanning forest is returned."""
    # TODO consider renaming this to minimum_spanning_forest
    pass  # pragma: no cover


@abstract_algorithm("traversal.astar_search")
def astar_search(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    source_node: NodeID,
    target_node: NodeID,
    heuristic_func: Callable[[NodeID], float],
) -> Vector:
    """
    heuristic_func is a unary function returning the distance between a given node and the target node.
    
    The return value is a vector of node ids representing the non-unique path found by the A* algorithm.
    """
    pass  # pragma: no cover
