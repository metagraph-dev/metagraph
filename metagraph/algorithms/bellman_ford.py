from metagraph import abstract_algorithm
from metagraph.types import Graph, Nodes
from typing import Tuple


@abstract_algorithm("traversal.bellman_ford")
def bellman_ford(graph: Graph(is_directed=True), source_node) -> Tuple[Nodes, Nodes]:
    pass
