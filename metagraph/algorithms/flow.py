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
    """
    Returns the maximum flow and a graph whose edge weights represent the flow.
    It contains all the nodes of the input graph
    """
    pass  # pragma: no cover


@abstract_algorithm("flow.min_cut")
def min_cut(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    source_node: NodeID,
    target_node: NodeID,
) -> Tuple[float, Graph]:
    """
    Returns the sum of the minimum cut weights and a graph containing only those edges
    which are part of the minimum cut.
    """
    pass  # pragma: no cover
