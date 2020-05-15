from metagraph import abstract_algorithm
from metagraph.types import Graph, Nodes
from typing import Tuple


@abstract_algorithm("clustering.connected_components")
def connected_components(graph: Graph(is_directed=False)) -> Nodes:
    pass  # pragma: no cover


@abstract_algorithm("clustering.strongly_connected_components")
def strongly_connected_components(graph: Graph(is_directed=True)) -> Nodes:
    pass  # pragma: no cover


@abstract_algorithm("clustering.label_propagation_community")
def label_propagation_community(graph: Graph(is_directed=False)) -> Nodes:
    pass  # pragma: no cover


@abstract_algorithm("clustering.louvain_community")
def louvain_community(graph: Graph(is_directed=False)) -> Tuple[Nodes, float]:
    pass  # pragma: no cover


@abstract_algorithm("cluster.triangle_count")
def triangle_count(graph: Graph(is_directed=False)) -> int:
    """
    Counts the number of unique triangles in an undirected graph
    """
    pass  # pragma: no cover
