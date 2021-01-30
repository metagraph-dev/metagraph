import metagraph as mg
from metagraph import abstract_algorithm
from ..types import Graph, NodeMap, NodeSet, NodeID
from typing import Tuple


@abstract_algorithm("centrality.betweenness")
def betweenness_centrality(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    nodes: mg.Optional[NodeSet] = None,
    normalize: bool = False,
) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("centrality.katz")
def katz_centrality(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    attenuation_factor: float = 0.01,
    immediate_neighbor_weight: float = 1.0,
    maxiter: int = 50,
    tolerance: float = 1e-05,
) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("centrality.pagerank")
def pagerank(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    damping: float = 0.85,
    maxiter: int = 50,
    tolerance: float = 1e-05,
) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("centrality.closeness")
def closeness_centrality(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    nodes: mg.Optional[NodeSet] = None,
) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("centrality.eigenvector")
def eigenvector_centrality(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}),
    maxiter: int = 50,
    tolerance: float = 1e-05,
) -> NodeMap:
    pass  # pragma: no cover


@abstract_algorithm("centrality.hits")
def hits_centrality(
    graph: Graph(edge_type="map", edge_dtype={"int", "float"}, is_directed=True),
    maxiter: int = 50,
    tolerance: float = 1e-05,
    normalize: bool = True,
) -> Tuple[NodeMap, NodeMap]:
    """
    Hyperlink-Induced Topic Search (HITS) algorithm

    Returns (hubs, authority)
    """
    pass  # pragma: no cover


@abstract_algorithm("centrality.degree")
def degree_centrality(
    graph: Graph, in_edges: bool = False, out_edges: bool = True,
) -> NodeMap:
    """The degree centrality of a node is node_degree / (total_num_nodes-1) ."""
    pass  # pragma: no cover
