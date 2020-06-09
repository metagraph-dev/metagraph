from metagraph import abstract_algorithm
from metagraph.types import EdgeSet, EdgeMap, NodeMap
from typing import Tuple


@abstract_algorithm("clustering.connected_components")
def connected_components(graph: EdgeSet(is_directed=False)) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("clustering.strongly_connected_components")
def strongly_connected_components(graph: EdgeSet(is_directed=True)) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("clustering.label_propagation_community")
def label_propagation_community(graph: EdgeMap(is_directed=False)) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("clustering.louvain_community")
def louvain_community(graph: EdgeMap(is_directed=False)) -> Tuple[NodeMap, float]:
    pass  # pragma: no cover


@abstract_algorithm("cluster.triangle_count")
def triangle_count(graph: EdgeSet(is_directed=False)) -> int:
    """
    Counts the number of unique triangles in an undirected graph
    """
    pass  # pragma: no cover
