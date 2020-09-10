import metagraph as mg
from metagraph import abstract_algorithm
from metagraph.types import Graph, NodeID, EdgeMap
from typing import Tuple


@abstract_algorithm("flow.max_flow")
def max_flow(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    source_node: NodeID,
    target_node: NodeID,
) -> Tuple[float, Graph]:
    """The returned graph is a graph whose edge weights represent the outward flow. It contains all the nodes of the input graph"""
    pass  # pragma: no cover
