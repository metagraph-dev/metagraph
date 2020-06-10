from metagraph import abstract_algorithm
from metagraph.types import NodeSet, EdgeSet, EdgeMap


@abstract_algorithm("subgraph.extract_edgemap")
def extract_edgemap(graph: EdgeMap, nodes: NodeSet) -> EdgeMap:
    pass


@abstract_algorithm("subgraph.extract_edgeset")
def extract_edgeset(graph: EdgeSet, nodes: NodeSet) -> EdgeSet:
    pass


@abstract_algorithm("subgraph.k_core")
def k_core(graph: EdgeMap, k: int) -> EdgeMap:
    pass


@abstract_algorithm("subgraph.k_core_unweighted")
def k_core_unweighted(graph: EdgeSet, k: int) -> EdgeSet:
    pass
