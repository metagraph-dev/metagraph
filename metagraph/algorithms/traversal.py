from metagraph import abstract_algorithm
from metagraph.types import Graph, Vector, NodeMap
from typing import Any, Tuple


@abstract_algorithm("traversal.bellman_ford")
def bellman_ford(graph: Graph, source_node: Any) -> Tuple[NodeMap, NodeMap]:
    pass


@abstract_algorithm("traversal.all_shortest_paths")
def all_shortest_paths(graph: Graph) -> Tuple[Graph, Graph]:
    pass


@abstract_algorithm("traversal.breadth_first_search")
def breadth_first_search(graph: Graph, source_node: Any) -> Vector:
    pass


@abstract_algorithm("traversal.dijkstra")
def dijkstra(
    graph: Graph, source_node: Any, max_path_length: float
) -> Tuple[NodeMap, NodeMap]:
    pass
