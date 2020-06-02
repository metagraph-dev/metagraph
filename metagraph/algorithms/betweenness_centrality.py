from metagraph import abstract_algorithm
from metagraph.types import EdgeMap, NodeMap


# TODO: this signature is too specific to the networkx implementation
@abstract_algorithm("vertex_ranking.betweenness_centrality")
def betweenness_centrality(
    graph: EdgeMap, k: int, enable_normalization: bool, include_endpoints: bool
) -> NodeMap:
    pass  # pragma: no cover
