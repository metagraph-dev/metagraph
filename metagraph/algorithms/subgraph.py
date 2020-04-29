from metagraph import abstract_algorithm
from metagraph.types import Graph
from typing import Iterable, Any


@abstract_algorithm("subgraph.extract_subgraph")
def extract_subgraph(graph: Graph, nodes: Iterable[Any]) -> Graph:
    pass
