from metagraph import abstract_algorithm
from metagraph.types import Graph


@abstract_algorithm("cluster.triangle_count")
def triangle_count(graph: Graph) -> int:
    """
    Counts the number of unique triangles in an undirected graph
    """
    pass
