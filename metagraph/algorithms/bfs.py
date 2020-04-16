from metagraph import abstract_algorithm
from metagraph.types import Graph, Vector


@abstract_algorithm("traversal.breadth_first_search")
def breadth_first_search(graph: Graph(is_directed=True), source_node: int) -> Vector:
    pass
