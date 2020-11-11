import metagraph as mg
from metagraph import abstract_algorithm
from metagraph.types import (
    NodeSet,
    NodeMap,
    Vector,
    Matrix,
    NodeID,
    EdgeSet,
    EdgeMap,
    Graph,
    GraphSageNodeEmbedding,
)
from typing import Any, Tuple, Callable


@abstract_algorithm("util.nodeset.choose_random")
def nodeset_choose_random(x: NodeSet, k: int) -> NodeSet:
    pass  # pragma: no cover


@abstract_algorithm("util.nodeset.from_vector")
def nodeset_from_vector(x: Vector) -> NodeSet:
    pass  # pragma: no cover


@abstract_algorithm("util.nodemap.sort")
def nodemap_sort(
    x: NodeMap, ascending: bool = True, limit: mg.Optional[int] = None
) -> Vector:
    pass  # pragma: no cover


@abstract_algorithm("util.nodemap.select")
def nodemap_select(x: NodeMap, nodes: NodeSet) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("util.nodemap.filter")
def nodemap_filter(x: NodeMap, func: Callable[[Any], bool]) -> NodeSet:
    pass  # pragma: no cover


@abstract_algorithm("util.nodemap.apply")
def nodemap_apply(x: NodeMap, func: Callable[[Any], Any]) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("util.nodemap.reduce")
def nodemap_reduce(x: NodeMap, func: Callable[[Any, Any], Any]) -> Any:
    pass  # pragma: no cover


@abstract_algorithm("util.edgemap.from_edgeset")
def edgemap_from_edgeset(edgeset: EdgeSet, default_value: Any) -> EdgeMap:
    pass  # pragma: no cover


@abstract_algorithm("util.graph.degree")
def graph_degree(
    graph: Graph, in_edges: bool = False, out_edges: bool = True,
) -> NodeMap:
    pass  # pragma: no cover


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
    pass  # pragma: no cover


@abstract_algorithm("util.graph.filter_edges")
def graph_filter_edges(
    graph: Graph(edge_type="map"), func: Callable[[Any], bool]
) -> Graph:
    """
    func takes the edge weight and returns a bool
    Edges are removed, but nodes are kept (this may result in isolate nodes).
    """
    pass  # pragma: no cover


@abstract_algorithm("util.graph.assign_uniform_weight")
def graph_assign_uniform_weight(
    graph: Graph, weight: Any = 1
) -> Graph(edge_type="map"):
    """
    Make all the edge weights of a (possibly unweighted) have uniform magnitude.
    """
    pass  # pragma: no cover


@abstract_algorithm("util.graph.build")
def graph_build(
    edges: mg.Union[EdgeSet, EdgeMap],
    nodes: mg.Optional[mg.Union[NodeSet, NodeMap]] = None,
) -> Graph:
    pass  # pragma: no cover


@abstract_algorithm("util.graph.collapse_by_label")
def graph_collapse_by_label(
    graph: Graph(is_directed=False),
    labels: NodeMap,
    aggregator: Callable[[Any, Any], Any],
) -> Graph:
    pass  # pragma: no cover


@abstract_algorithm("util.graph.isomorphic")
def graph_isomorphic(g1: Graph, g2: Graph) -> bool:
    pass  # pragma: no cover


@abstract_algorithm("util.node_embedding.apply")
def node_embedding_apply(matrix: Matrix, node2row: NodeMap, nodes: Vector) -> Matrix:
    pass  # pragma: no cover


@abstract_algorithm("util.graph_sage_node_embedding.apply")
def graph_sage_node_embedding_apply(
    embedding: GraphSageNodeEmbedding,
    graph: Graph,
    node_features: Matrix,
    node2row: NodeMap,
) -> Matrix:
    pass  # pragma: no cover
