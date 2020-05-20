from metagraph import abstract_algorithm
from metagraph.types import Graph, NodeMap


@abstract_algorithm("link_analysis.pagerank")
def pagerank(
    graph: Graph, damping: float = 0.85, maxiter: int = 50, tolerance: float = 1e-05
) -> NodeMap:
    pass  # pragma: no cover
