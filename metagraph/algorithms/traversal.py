from metagraph import abstract_algorithm
from metagraph.types import Graph, Vector, Nodes
from typing import Any, Tuple


@abstract_algorithm("traversal.bellman_ford")
def bellman_ford(
    graph: Graph(is_directed=True), source_node: Any
) -> Tuple[Nodes, Nodes]:
    pass


@abstract_algorithm("traversal.breadth_first_search")
def breadth_first_search(graph: Graph(is_directed=True), source_node: Any) -> Vector:
    pass


@abstract_algorithm("traversal.dijkstra")
def dijkstra(
    graph: Graph, source_node: Any, max_path_length: float
) -> Tuple[Nodes, Nodes]:
    pass
