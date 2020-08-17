from metagraph import abstract_algorithm
from metagraph.types import NodeSet, Graph


@abstract_algorithm("subgraph.extract_subgraph")
def extract_subgraph(graph: Graph, nodes: NodeSet) -> Graph:
    pass  # pragma: no cover


@abstract_algorithm("subgraph.k_core")
def k_core(graph: Graph(is_directed=False), k: int) -> Graph:
    pass  # pragma: no cover
