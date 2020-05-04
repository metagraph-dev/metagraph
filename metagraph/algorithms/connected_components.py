from metagraph import abstract_algorithm
from metagraph.types import Graph, Nodes


@abstract_algorithm("clustering.connected_components")
def connected_components(graph: Graph(is_directed=False)) -> Nodes:
    pass  # pragma: no cover


@abstract_algorithm("clustering.strongly_connected_components")
def strongly_connected_components(graph: Graph(is_directed=True)) -> Nodes:
    pass  # pragma: no cover
