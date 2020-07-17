from metagraph import abstract_algorithm
from metagraph.types import NodeSet, NodeMap, Vector, NodeID  # Graph
from typing import Any, Callable, Optional


@abstract_algorithm("util.nodeset.choose_random")
def nodeset_choose_random(x: NodeSet, k: int) -> NodeSet:
    pass


@abstract_algorithm("util.nodemap.sort")
def nodemap_sort(
    x: NodeMap, ascending: bool = True, limit: Optional[int] = None
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


# @abstract_algorithm("util.graph.aggregate_edges")
# def graph_aggregate_edges(graph: Graph(edge_type="map"), func: Callable[[Any, Any], Any], in_edges=False, out_edges=True) -> NodeMap:
#    pass

# @abstract_algorithm("util.graph.filter_edges")
# def graph_filter_edges(graph: Graph, filter: Callable[[Any], bool]) -> Graph:
#    pass

# @abstract_algorithm("util.graph.add_uniform_weight")
# def graph_add_uniform_weight(graph: Graph(edge_type="set"), weight: Any = 1) -> Graph(edge_type="map"):
#    pass
