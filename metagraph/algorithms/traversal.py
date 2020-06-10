from metagraph import abstract_algorithm
from metagraph.types import EdgeSet, EdgeMap, Vector, NodeMap, NodeID
from typing import Any, Tuple


@abstract_algorithm("traversal.bellman_ford")
def bellman_ford(graph: EdgeMap, source_node: NodeID) -> Tuple[NodeMap, NodeMap]:
    pass


@abstract_algorithm("traversal.all_shortest_paths")
def all_shortest_paths(graph: EdgeMap) -> Tuple[EdgeMap, EdgeMap]:
    pass


@abstract_algorithm("traversal.breadth_first_search")
def breadth_first_search(graph: EdgeSet, source_node: NodeID) -> Vector:
    pass


@abstract_algorithm("traversal.dijkstra")
def dijkstra(
    graph: EdgeMap(has_negative_weights=False),
    source_node: NodeID,
    max_path_length: float,
) -> Tuple[NodeMap, NodeMap]:
    pass
