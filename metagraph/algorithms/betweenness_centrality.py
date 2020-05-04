from metagraph import abstract_algorithm
from metagraph.types import Graph, Nodes


# TODO: this signature is too specific to the networkx implementation
@abstract_algorithm("vertex_ranking.betweenness_centrality")
def betweenness_centrality(
    graph: Graph, k: int, enable_normalization: bool, include_endpoints: bool
) -> Nodes:
    pass
