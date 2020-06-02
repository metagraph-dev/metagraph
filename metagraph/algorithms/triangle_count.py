from metagraph import abstract_algorithm
from metagraph.types import EdgeMap, NodeMap


@abstract_algorithm("cluster.triangle_count")
def triangle_count(graph: EdgeMap(is_directed=False)) -> int:
    """
    Counts the number of unique triangles in an undirected graph
    """
    pass  # pragma: no cover
