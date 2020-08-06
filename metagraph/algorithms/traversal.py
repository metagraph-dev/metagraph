from metagraph import abstract_algorithm
from metagraph.types import Graph, Vector, NodeMap, NodeID
from typing import Tuple


@abstract_algorithm("traversal.bellman_ford")
def bellman_ford(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}), source_node: NodeID
) -> Tuple[NodeMap, NodeMap]:
    """Output is (Parents, Distance)"""
    pass


@abstract_algorithm("traversal.all_pairs_shortest_paths")
def all_pairs_shortest_paths(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"})
) -> Tuple[Graph, Graph]:
    """Output is (Parents, Distance)"""
    pass


@abstract_algorithm("traversal.bfs_iter")
def breadth_first_search_iterator(
    graph: Graph, source_node: NodeID, depth_limit: int = -1
) -> Vector:
    """Output is NodeID"""
    pass


@abstract_algorithm("traversal.bfs_tree")
def breadth_first_search_tree(
    graph: Graph, source_node: NodeID, depth_limit: int = -1
) -> Tuple[NodeMap, NodeMap]:
    """Output is (Depth, Parents)"""
    pass


@abstract_algorithm("traversal.dijkstra")
def dijkstra(
    graph: Graph(
        edge_has_negative_weights=False, edge_type="map", edge_dtype={"int", "float"}
    ),
    source_node: NodeID,
) -> Tuple[NodeMap, NodeMap]:
    """Output is (Parents, Distance)"""
    pass
