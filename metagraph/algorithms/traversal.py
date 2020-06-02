from metagraph import abstract_algorithm
from metagraph.types import EdgeMap, Vector, NodeMap
from typing import Any, Tuple


@abstract_algorithm("traversal.bellman_ford")
def bellman_ford(graph: EdgeMap, source_node: Any) -> Tuple[NodeMap, NodeMap]:
    pass


@abstract_algorithm("traversal.all_shortest_paths")
def all_shortest_paths(graph: EdgeMap) -> Tuple[EdgeMap, EdgeMap]:
    pass


@abstract_algorithm("traversal.breadth_first_search")
def breadth_first_search(graph: EdgeMap, source_node: Any) -> Vector:
    pass


@abstract_algorithm("traversal.dijkstra")
def dijkstra(
    graph: EdgeMap, source_node: Any, max_path_length: float
) -> Tuple[NodeMap, NodeMap]:
    pass
