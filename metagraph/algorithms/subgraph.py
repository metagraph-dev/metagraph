from metagraph import abstract_algorithm
from metagraph.types import Graph, Vector


@abstract_algorithm("subgraph.extract_subgraph")
def extract_subgraph(graph: Graph, nodes: Vector) -> Graph:
    pass


@abstract_algorithm("subgraph.k_core")
def k_core(graph: Graph, k: int) -> Graph:
    pass
