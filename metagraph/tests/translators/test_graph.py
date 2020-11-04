import pytest

grblas = pytest.importorskip("grblas")

from metagraph.tests.util import default_plugin_resolver
from . import RoundTripper
from metagraph.plugins.numpy.types import NumpyNodeSet
from metagraph.plugins.scipy.types import ScipyEdgeMap, ScipyEdgeSet, ScipyGraph
from metagraph.plugins.networkx.types import NetworkXGraph
from metagraph.plugins.graphblas.types import GrblasEdgeMap
from metagraph.plugins.pandas.types import PandasEdgeSet
from metagraph import NodeLabels
import networkx as nx
import scipy.sparse as ss
import pandas as pd
import numpy as np


def test_graph_roundtrip_directed_unweighted(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.DiGraph()
    g.add_nodes_from([1, 3, 5, 7, 8, 9, 10, 11, 15])
    g.add_edges_from([(1, 3), (3, 1), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)])
    graph = NetworkXGraph(g)
    rt.verify_round_trip(graph)


def test_graph_roundtrip_directed_weighted(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.DiGraph()
    g.add_nodes_from([1, 3, 5, 7, 8, 9, 10, 11, 15])
    edges = [(1, 3), (3, 1), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)]
    edge_weights = [1.1, 2.2, 0.0, -4.4, 4.4, 6.5, 1.2, 2.0]
    # float with neg weights
    g.add_weighted_edges_from(
        [(src, dst, wgt) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # float without neg weights
    g.add_weighted_edges_from(
        [(src, dst, abs(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # int with neg weights
    g.add_weighted_edges_from(
        [(src, dst, int(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # int without neg weights
    g.add_weighted_edges_from(
        [(src, dst, abs(int(wgt))) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # bool
    g.add_weighted_edges_from(
        [(src, dst, bool(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))


def test_graph_roundtrip_directed_symmetric(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.DiGraph()
    g.add_nodes_from([1, 3, 5, 7, 8, 9, 10, 11, 15])
    edges = [(1, 3), (3, 1), (3, 5), (5, 3), (3, 9), (9, 3), (5, 5), (11, 10), (10, 11)]
    edge_weights = [1.1, 1.1, 0.0, -4.4, 4.4, 6.5, 1.2, 2.0]
    # float with neg weights
    g.add_weighted_edges_from(
        [(src, dst, wgt) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # float without neg weights
    g.add_weighted_edges_from(
        [(src, dst, abs(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # int with neg weights
    g.add_weighted_edges_from(
        [(src, dst, int(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # int without neg weights
    g.add_weighted_edges_from(
        [(src, dst, abs(int(wgt))) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # bool
    g.add_weighted_edges_from(
        [(src, dst, bool(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))


def test_graph_roundtrip_undirected_unweighted(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.Graph()
    g.add_nodes_from([1, 3, 5, 7, 8, 9, 10, 11, 15])
    g.add_edges_from([(1, 3), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)])
    graph = NetworkXGraph(g)
    rt.verify_round_trip(graph)


def test_graph_roundtrip_undirected_weighted(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.Graph()
    g.add_nodes_from([1, 3, 5, 7, 8, 9, 10, 11, 15])
    edges = [(1, 3), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)]
    edge_weights = [1.1, 0.0, -4.4, 4.4, 6.5, 1.2, 2.0]
    # float with neg weights
    g.add_weighted_edges_from(
        [(src, dst, wgt) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # float without neg weights
    g.add_weighted_edges_from(
        [(src, dst, abs(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # int with neg weights
    g.add_weighted_edges_from(
        [(src, dst, int(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # int without neg weights
    g.add_weighted_edges_from(
        [(src, dst, abs(int(wgt))) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # bool
    g.add_weighted_edges_from(
        [(src, dst, bool(wgt)) for (src, dst), wgt in zip(edges, edge_weights)]
    )
    rt.verify_round_trip(NetworkXGraph(g))


def test_graph_roundtrip_directed_unweighted_nodevals(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.DiGraph()
    g.add_edges_from([(1, 3), (3, 1), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)])
    nodes = [1, 3, 5, 7, 8, 9, 10, 11, 15]
    node_weights = [1.1, 0.0, -4.4, 4.4, 6.5, 1.2, 2.0, 0.01, 15.2]
    g.add_nodes_from(nodes)
    # nodevals as floats
    nx.set_node_attributes(
        g, {node: wgt for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # nodevals as ints
    nx.set_node_attributes(
        g, {node: int(wgt) for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # nodevals as bools
    nx.set_node_attributes(
        g, {node: bool(wgt) for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))


def test_graph_roundtrip_directed_weighted_nodevals(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.DiGraph()
    nodes = [1, 3, 5, 7, 8, 9, 10, 11, 15]
    node_weights = [1.1, 0.0, -4.4, 4.4, 6.5, 1.2, 2.0, 0.01, 15.2]
    edges = [(1, 3), (3, 1), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)]
    edge_weights = [1.1, 2.2, 0.0, -4.4, 4.4, 6.5, 1.2, 2.0]
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    # nodevals as floats, edges as ints
    nx.set_node_attributes(
        g, {node: wgt for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    nx.set_edge_attributes(
        g, {edge: int(wgt) for edge, wgt in zip(edges, edge_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # nodevals as ints, edges as bools
    nx.set_node_attributes(
        g, {node: int(wgt) for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    nx.set_edge_attributes(
        g, {edge: bool(wgt) for edge, wgt in zip(edges, edge_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # nodevals as bools, edges as floats
    nx.set_node_attributes(
        g, {node: bool(wgt) for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    nx.set_edge_attributes(
        g, {edge: wgt for edge, wgt in zip(edges, edge_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))


def test_graph_roundtrip_undirected_unweighted_nodevals(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.Graph()
    g.add_edges_from([(1, 3), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)])
    nodes = [1, 3, 5, 7, 8, 9, 10, 11, 15]
    node_weights = [1.1, 0.0, -4.4, 4.4, 6.5, 1.2, 2.0, 0.01, 15.2]
    g.add_nodes_from(nodes)
    # nodevals as floats
    nx.set_node_attributes(
        g, {node: wgt for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # nodevals as ints
    nx.set_node_attributes(
        g, {node: int(wgt) for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # nodevals as bools
    nx.set_node_attributes(
        g, {node: bool(wgt) for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))


def test_graph_roundtrip_undirected_weighted_nodevals(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.Graph()
    nodes = [1, 3, 5, 7, 8, 9, 10, 11, 15]
    node_weights = [1.1, 0.0, -4.4, 4.4, 6.5, 1.2, 2.0, 0.01, 15.2]
    edges = [(1, 3), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)]
    edge_weights = [1.1, 0.0, -4.4, 4.4, 6.5, 1.2, 2.0]
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    # nodevals as floats, edges as bools
    nx.set_node_attributes(
        g, {node: wgt for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    nx.set_edge_attributes(
        g, {edge: bool(wgt) for edge, wgt in zip(edges, edge_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # nodevals as ints, edges as floats
    nx.set_node_attributes(
        g, {node: int(wgt) for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    nx.set_edge_attributes(
        g, {edge: wgt for edge, wgt in zip(edges, edge_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))
    # nodevals as bools, edges as ints
    nx.set_node_attributes(
        g, {node: bool(wgt) for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    nx.set_edge_attributes(
        g, {edge: int(wgt) for edge, wgt in zip(edges, edge_weights)}, name="weight"
    )
    rt.verify_round_trip(NetworkXGraph(g))


def test_graph_edgeset_oneway_directed(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.DiGraph()
    g.add_nodes_from([1, 3, 5, 7, 8, 9, 10, 11, 15])
    g.add_edges_from([(1, 3), (3, 1), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)])
    graph = NetworkXGraph(g)
    df = pd.DataFrame(
        {"source": [3, 3, 1, 5, 7, 9, 11, 5], "target": [5, 1, 3, 7, 9, 3, 10, 5]}
    )
    edgeset = PandasEdgeSet(df, is_directed=True)
    rt.verify_one_way(graph, edgeset)


def test_graph_edgeset_oneway_directed_symmetric(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.DiGraph()
    g.add_nodes_from([1, 3, 5, 7, 8, 9, 10, 11, 15])
    g.add_edges_from(
        [(1, 3), (3, 1), (3, 5), (5, 3), (3, 9), (9, 3), (5, 5), (11, 10), (10, 11)]
    )
    graph = NetworkXGraph(g)
    df = pd.DataFrame(
        {
            "source": [1, 3, 3, 5, 3, 9, 5, 11, 10],
            "target": [3, 1, 5, 3, 9, 3, 5, 10, 11],
        }
    )
    edgeset = PandasEdgeSet(df, is_directed=True)
    rt.verify_one_way(graph, edgeset)


def test_graph_edgeset_oneway_undirected(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.Graph()
    g.add_nodes_from([1, 3, 5, 7, 8, 9, 10, 11, 15])
    g.add_edges_from([(1, 3), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)])
    graph = NetworkXGraph(g)
    df = pd.DataFrame(
        {"source": [11, 1, 3, 5, 7, 9, 5], "target": [10, 3, 5, 7, 9, 3, 5]}
    )
    edgeset = PandasEdgeSet(df, is_directed=False)
    rt.verify_one_way(graph, edgeset)


def test_graph_nodeset_oneway(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    g = nx.Graph()
    nodes = [1, 3, 5, 7, 8, 9, 10, 11, 15]
    node_weights = [1.1, 0.0, -4.4, 4.4, 6.5, 1.2, 2.0, 0.01, 15.2]
    edges = [(1, 3), (3, 5), (5, 7), (7, 9), (9, 3), (5, 5), (11, 10)]
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    nx.set_node_attributes(
        g, {node: wgt for node, wgt in zip(nodes, node_weights)}, name="weight"
    )
    rt.verify_one_way(NetworkXGraph(g), NumpyNodeSet(nodes))


def test_networkx_scipy_graph_from_edgemap(default_plugin_resolver):
    dpr = default_plugin_resolver
    g = nx.DiGraph()
    g.add_weighted_edges_from([(2, 2, 1), (2, 7, 2), (7, 7, 0), (7, 0, 3), (0, 7, 3)])
    x = NetworkXGraph(g)
    # Convert networkx -> scipy adjacency
    #    0 2 7
    # 0 [    3]
    # 2 [  1 2]
    # 7 [3   0]
    m = ss.coo_matrix(
        ([3, 1, 2, 3, 0], ([0, 1, 1, 2, 2], [2, 1, 2, 0, 2])), dtype=np.int64
    )
    intermediate = ScipyGraph(m, [0, 2, 7])
    y = dpr.translate(x, ScipyGraph)
    dpr.assert_equal(y, intermediate)


def test_networkx_scipy_graph_from_edgeset(default_plugin_resolver):
    dpr = default_plugin_resolver
    g = nx.DiGraph()
    g.add_edges_from([(2, 2), (2, 7), (7, 7), (7, 0), (0, 7)])
    x = NetworkXGraph(g)
    # Convert networkx -> scipy adjacency
    #    0 2 7
    # 0 [    1]
    # 2 [  1 1]
    # 7 [1   1]
    m = ss.coo_matrix(([1, 1, 1, 1, 1], ([0, 1, 1, 2, 2], [2, 1, 2, 0, 2])), dtype=bool)
    intermediate = ScipyGraph(m, [0, 2, 7])
    y = dpr.translate(x, ScipyGraph)
    dpr.assert_equal(y, intermediate)


def test_scipy_graphblas_edgemap(default_plugin_resolver):
    dpr = default_plugin_resolver
    #    0 2 7
    # 0 [1 2  ]
    # 2 [  0 3]
    # 7 [  3  ]
    g = ss.coo_matrix(
        ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1])), dtype=np.int64
    )
    x = ScipyEdgeMap(g, [0, 2, 7])
    # Convert scipy adjacency to graphblas
    m = grblas.Matrix.from_values(
        [0, 0, 2, 2, 7], [0, 2, 2, 7, 2], [1, 2, 0, 3, 3], dtype=grblas.dtypes.INT64
    )
    intermediate = GrblasEdgeMap(m)
    y = dpr.translate(x, GrblasEdgeMap)
    dpr.assert_equal(y, intermediate)


# def test_networkx_2_pandas(default_plugin_resolver):
#     dpr = default_plugin_resolver
#     g = nx.DiGraph()
#     g.add_weighted_edges_from([(2, 2, 1), (2, 7, 2), (7, 7, 0), (7, 0, 3), (0, 7, 3)])
#     x = NetworkXGraph(g)
#     # Convert networkx -> pandas edge list
#     df = pd.DataFrame(
#         {
#             "source": [2, 2, 7, 0, 7],
#             "target": [2, 7, 0, 7, 7],
#             "weight": [1, 2, 3, 3, 0],
#         }
#     )
#     intermediate = PandasEdgeMap(df, weight_label="weight")
#     y = dpr.translate(x, PandasEdgeMap)
#     dpr.assert_equal(y, intermediate)
#     # Convert networkx <- pandas edge list
#     x2 = dpr.translate(y, NetworkXGraph)
#     dpr.assert_equal(x, x2)
