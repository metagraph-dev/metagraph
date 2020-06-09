from metagraph import abstract_algorithm
from metagraph.types import EdgeMap, Vector


@abstract_algorithm("subgraph.extract_subgraph")
def extract_subgraph(graph: EdgeMap, nodes: Vector) -> EdgeMap:
    pass


@abstract_algorithm("subgraph.k_core")
def k_core(graph: EdgeMap, k: int) -> EdgeMap:
    pass
