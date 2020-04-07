from metagraph import abstract_algorithm
from metagraph.types import Graph, Nodes


@abstract_algorithm("cluster.triangle_count")
def triangle_count(graph: Graph(is_directed=False)) -> int:
    """
    Counts the number of unique triangles in an undirected graph
    """
    pass


@abstract_algorithm("cluster.triangle_count_by_node")
def triangle_count_by_node(
    graph: Graph(is_directed=False),
) -> Nodes(weights="positive"):
    """
    Count the number of triangles each node is a part of
    Nodes which are not part of any triangles will be missing rather than have a value of 0
    """
    pass
