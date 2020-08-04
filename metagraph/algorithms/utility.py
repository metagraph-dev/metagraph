import metagraph as mg
from metagraph import abstract_algorithm
from metagraph.types import NodeSet, NodeMap, Vector, NodeID, EdgeSet, EdgeMap, Graph
from typing import Any, Callable


@abstract_algorithm("util.nodeset.choose_random")
def nodeset_choose_random(x: NodeSet, k: int) -> NodeSet:
    pass


@abstract_algorithm("util.nodemap.sort")
def nodemap_sort(
    x: NodeMap, ascending: bool = True, limit: mg.Optional[int] = None
) -> Vector:
    pass


@abstract_algorithm("util.nodemap.select")
def nodemap_select(x: NodeMap, nodes: NodeSet) -> NodeMap:
    pass


@abstract_algorithm("util.nodemap.filter")
def nodemap_filter(x: NodeMap, func: Callable[[Any], bool]) -> NodeSet:
    pass


@abstract_algorithm("util.nodemap.apply")
def nodemap_apply(x: NodeMap, func: Callable[[Any], Any]) -> NodeMap:
    pass


@abstract_algorithm("util.nodemap.reduce")
def nodemap_reduce(x: NodeMap, func: Callable[[Any, Any], Any]) -> Any:
    pass


@abstract_algorithm("util.graph.aggregate_edges")
def graph_aggregate_edges(
    graph: Graph(edge_type="map"),
    func: Callable[[Any, Any], Any],
    initial_value: Any,
    in_edges: bool = False,
    out_edges: bool = True,
) -> NodeMap:
    """
    if in_edges == out_edges == False, every node is mapped to initial_value 
    if the graph is undirected and either in_edges == True or out_edges == True, we aggregate over all of the edges for each node exactly once to avoid double counting
    if the graph is directed, in_edges and out_edges specify which edge types to aggregate over for a given node
    """
    pass


@abstract_algorithm("util.graph.filter_edges")
def graph_filter_edges(
    graph: Graph(edge_type="map"), func: Callable[[Any], bool]
) -> Graph:
    """
    func takes the edge weight and returns a bool
    Edges are removed, but nodes are kept (this may result in orphan nodes).
    """
    pass


@abstract_algorithm("util.graph.assign_uniform_weight")
def graph_assign_uniform_weight(
    graph: Graph(edge_type="set"), weight: Any = 1
) -> Graph(edge_type="map"):
    """
    Gives an unweighted graph weights of uniform magnitude.
    """
    pass


@abstract_algorithm("util.graph.build")
def graph_build(
    edges: mg.Union[EdgeSet, EdgeMap],
    nodes: mg.Optional[mg.Union[NodeSet, NodeMap]] = None,
) -> Graph:
    pass
