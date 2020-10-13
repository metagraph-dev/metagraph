from metagraph import abstract_algorithm
from metagraph.types import Graph, NodeMap
from typing import Tuple, Callable, Any


@abstract_algorithm("clustering.connected_components")
def connected_components(graph: Graph(is_directed=False)) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("clustering.strongly_connected_components")
def strongly_connected_components(graph: Graph(is_directed=True)) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("clustering.label_propagation_community")
def label_propagation_community(graph: Graph(is_directed=False)) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("clustering.louvain_community")
def louvain_community_step(
    graph: Graph(is_directed=False, edge_type="map", edge_dtype={"int", "float"})
) -> Tuple[NodeMap, float]:
    """Runs one step of louvain, returning communities and modularity score"""
    pass  # pragma: no cover


# TODO: why is this "cluster" instead of "clustering"?
@abstract_algorithm("cluster.triangle_count")
def triangle_count(graph: Graph(is_directed=False)) -> int:
    """Counts the number of unique triangles in an undirected graph"""
    pass  # pragma: no cover


@abstract_algorithm("clustering.coloring.greedy")
def greedy_coloring(graph: Graph(is_directed=False)) -> Tuple[NodeMap, int]:
    """
    Attempts to find the minimum number of colors required to color the graph such that no connected
    nodes have the same color. Color is simply represented as a value from 0..n

    Returns color for each node and # of colors required
    """
    pass  # pragma: no covert
