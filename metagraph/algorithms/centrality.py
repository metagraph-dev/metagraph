import metagraph as mg
from metagraph import abstract_algorithm
from metagraph.types import Graph, NodeSet, NodeMap, NodeID


@abstract_algorithm("centrality.betweenness")
def betweenness_centrality(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    nodes: mg.Optional[NodeSet] = None,
    normalize: bool = False,
) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("centrality.katz")
def katz_centrality(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    attenuation_factor: float = 0.01,
    immediate_neighbor_weight: float = 1.0,
    maxiter: int = 50,
    tolerance: float = 1e-05,
) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("centrality.pagerank")
def pagerank(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    damping: float = 0.85,
    maxiter: int = 50,
    tolerance: float = 1e-05,
) -> NodeMap:
    pass  # pragma: no cover
