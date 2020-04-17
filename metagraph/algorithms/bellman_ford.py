from metagraph import abstract_algorithm
from metagraph.types import Graph, Nodes
from typing import Any, Tuple


@abstract_algorithm("traversal.bellman_ford")
def bellman_ford(
    graph: Graph(is_directed=True), source_node: Any
) -> Tuple[Nodes, Nodes]:
    pass
