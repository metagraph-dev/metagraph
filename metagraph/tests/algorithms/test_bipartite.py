from metagraph.tests.util import default_plugin_resolver
import networkx as nx
from . import MultiVerify


def test_graph_projection(default_plugin_resolver):
    r"""
    0  1  2  3
    |\   /|\  \
    | \ / | \  \
    5  6  7  8  9
    """
    dpr = default_plugin_resolver
    ebunch = [
        (0, 5),
        (0, 6),
        (2, 6),
        (2, 7),
        (2, 8),
        (3, 9),
    ]
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(ebunch)
    bgraph = dpr.wrappers.BipartiteGraph.NetworkXBipartiteGraph(
        nx_graph, [(0, 1, 2, 3), (5, 6, 7, 8, 9)]
    )
    projected_edges = [
        (5, 6),
        (6, 7),
        (6, 8),
        (7, 8),
    ]
    proj_graph = nx.Graph()
    proj_graph.add_nodes_from([5, 6, 7, 8, 9])  # needed because 9 is an isolate node
    proj_graph.add_edges_from(projected_edges)
    result = dpr.wrappers.Graph.NetworkXGraph(proj_graph)
    MultiVerify(dpr).compute("bipartite.graph_projection", bgraph, 1).assert_equal(
        result
    )
