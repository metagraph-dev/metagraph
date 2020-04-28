from metagraph import abstract_algorithm
from metagraph.types import Graph, Vector
from typing import Any


@abstract_algorithm("traversal.breadth_first_search")
def breadth_first_search(graph: Graph(is_directed=True), source_node: Any) -> Vector:
    pass
