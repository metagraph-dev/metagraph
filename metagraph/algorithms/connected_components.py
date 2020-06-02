from metagraph import abstract_algorithm
from metagraph.types import EdgeMap, NodeMap


@abstract_algorithm("clustering.connected_components")
def connected_components(graph: EdgeMap(is_directed=False)) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("clustering.strongly_connected_components")
def strongly_connected_components(graph: EdgeMap(is_directed=True)) -> NodeMap:
    pass  # pragma: no cover
