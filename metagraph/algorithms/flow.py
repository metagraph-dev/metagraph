import metagraph as mg
from metagraph import abstract_algorithm
from metagraph.types import Graph, NodeID, EdgeMap


@abstract_algorithm("flow.max_flow")
def max_flow(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    source_node: NodeID,
    target_node: NodeID,
) -> EdgeMap:
    pass  # pragma: no cover
