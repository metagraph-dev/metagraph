from metagraph import abstract_algorithm
from metagraph.types import Graph, NodeMap


@abstract_algorithm("cluster.triangle_count")
def triangle_count(graph: Graph(is_directed=False)) -> int:
    """
    Counts the number of unique triangles in an undirected graph
    """
    pass  # pragma: no cover
