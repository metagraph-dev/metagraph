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


@abstract_algorithm("util.graph.collapse_by_label")
def collapse_by_label(
    graph: Graph(is_directed=False),
    labels: NodeMap,
    aggregator: Callable[[Any, Any], Any],
) -> Graph:
    pass


@abstract_algorithm("cluster.triangle_count")
def triangle_count(graph: Graph(is_directed=False)) -> int:
    """Counts the number of unique triangles in an undirected graph"""
    pass  # pragma: no cover
