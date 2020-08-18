from metagraph import abstract_algorithm
from metagraph.types import Graph, BipartiteGraph


@abstract_algorithm("bipartite.graph_projection")
def graph_projection(bgraph: BipartiteGraph, nodes_retained: int = 0) -> Graph:
    pass  # pragma: no cover
