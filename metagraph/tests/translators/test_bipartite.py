import pytest
from metagraph.tests.util import default_plugin_resolver
from metagraph.plugins.networkx.types import NetworkXBipartiteGraph
from . import RoundTripper
import networkx as nx


def test_bipartitegraph_roundtrip_undirected_unweighted(default_plugin_resolver):
    dpr = default_plugin_resolver
    rt = RoundTripper(dpr)
    r"""
    0  1  2  3
    |\   /|\  \
    | \ / | \  \
    5  6  7  8  9
    """
    edges = [(0, 5), (0, 6), (2, 6), (2, 7), (2, 8), (3, 9)]
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edges)
    bgraph = dpr.wrappers.BipartiteGraph.NetworkXBipartiteGraph(
        nx_graph, [(0, 1, 2, 3), (5, 6, 7, 8, 9)]
    )
    rt.verify_round_trip(bgraph)


def test_bipartitegraph_roundtrip_undirected_weighted(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    r"""
    0  1  2  3
    |\   /|\  \
    | \ / | \  \
    5  6  7  8  9
    """
    edges = [(0, 5), (0, 6), (2, 6), (2, 7), (2, 8), (3, 9)]
    edge_weights = [1.1, -2.2, 0.0, 2.7, 3.3, 0.0]
    clusters = [(0, 1, 2, 3), (5, 6, 7, 8, 9)]
    g = nx.Graph()
    # float with neg weights
    g.add_weighted_edges_from(
        [(src, dst, wgt) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXBipartiteGraph(g, clusters))
    # float without neg weights
    g.add_weighted_edges_from(
        [(src, dst, abs(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXBipartiteGraph(g, clusters))
    # int with neg weights
    g.add_weighted_edges_from(
        [(src, dst, int(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXBipartiteGraph(g, clusters))
    # int without neg weights
    g.add_weighted_edges_from(
        [(src, dst, abs(int(wgt))) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXBipartiteGraph(g, clusters))
    # bool
    g.add_weighted_edges_from(
        [(src, dst, bool(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXBipartiteGraph(g, clusters))


# TODO: add more tests once there are more concrete implementations of bipartite graph
