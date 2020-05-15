from metagraph import abstract_algorithm
from metagraph.types import Graph, Nodes


# TODO: this signature is too specific to the networkx implementation
@abstract_algorithm("vertex_ranking.betweenness_centrality")
def betweenness_centrality(
    graph: Graph, k: int, enable_normalization: bool, include_endpoints: bool
) -> Nodes:
    pass  # pragma: no cover


@abstract_algorithm("link_analysis.katz_centrality")
def katz_centrality(
    graph: Graph,
    attenuation_factor: float = 0.01,
    immediate_neighbor_weight: float = 1.0,
    maxiter: int = 50,
    tolerance: float = 1e-05,
) -> Nodes:
    pass  # pragma: no cover


@abstract_algorithm("link_analysis.pagerank")
def pagerank(
    graph: Graph, damping: float = 0.85, maxiter: int = 50, tolerance: float = 1e-05
) -> Nodes:
    pass  # pragma: no cover
