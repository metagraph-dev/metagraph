import pytest
from metagraph.tests.util import default_plugin_resolver
import networkx as nx
import numpy as np
from . import MultiVerify
from metagraph.plugins.networkx.types import NetworkXGraph


def build_standard_graph():
    r"""
    0 <--2-- 1        5 --10-> 6
    |      ^ |      ^ ^      /
    |     /  |     /  |     /
    1    7   3    9   5   11
    |   /    |  /     |   /
    v        v /        v
    3 --8--> 4 <--4-- 2 --6--> 7
    """
    ebunch = [
        (0, 3, 1),
        (1, 0, 2),
        (1, 4, 3),
        (2, 4, 4),
        (2, 5, 5),
        (2, 7, 6),
        (3, 1, 7),
        (3, 4, 8),
        (4, 5, 9),
        (5, 6, 10),
        (6, 2, 11),
    ]
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(ebunch)
    return NetworkXGraph(nx_graph)


def test_betweenness_centrality_single_hub(default_plugin_resolver):
    dpr = default_plugin_resolver
    graph = build_standard_graph()
    nodes = dpr.wrappers.NodeSet.PythonNodeSet({0, 1, 2, 3, 4, 5, 6, 7})
    expected_answer_unwrapped = {
        0: 1.0,
        1: 1.0,
        2: 9.0,
        3: 6.0,
        4: 12.0,
        5: 13.0,
        6: 11.0,
        7: 0.0,
    }
    expected_answer = dpr.wrappers.NodeMap.PythonNodeMap(expected_answer_unwrapped)
    MultiVerify(dpr).compute(
        "centrality.betweenness", graph, nodes, normalize=False,
    ).assert_equal(expected_answer)
    MultiVerify(dpr).compute(
        "centrality.betweenness", graph, normalize=False,
    ).assert_equal(expected_answer)


def test_betweenness_centrality_multiple_hubs(default_plugin_resolver):
    r"""
    0 --10-> 1 --1--> 5 --10-> 6
    |      ^ ^        ^      /
    0.1   /  |        |     /
    |    10  10       5   11
    |  _/    |        |   /
    v /      |          v
    3 -0.1-> 4 --1--> 2 --6--> 7
    """
    dpr = default_plugin_resolver
    ebunch = [
        (0, 1, 2),
        (0, 3, 0.1),
        (1, 5, 1),
        (2, 5, 5),
        (2, 7, 6),
        (3, 1, 7),
        (3, 4, 0.1),
        (4, 1, 3),
        (4, 2, 1),
        (5, 6, 10),
        (6, 2, 11),
    ]
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
    nodes = dpr.wrappers.NodeSet.PythonNodeSet({0, 1, 2, 3, 4, 5, 6, 7})
    expected_answer_unwrapped = {
        0: 0.0,
        1: 6.0,
        2: 7.0,
        3: 3.0,
        4: 7.0,
        5: 7.0,
        6: 4.0,
        7: 0.0,
    }
    expected_answer = dpr.wrappers.NodeMap.PythonNodeMap(expected_answer_unwrapped)
    MultiVerify(dpr).compute(
        "centrality.betweenness", graph, nodes, normalize=False,
    ).assert_equal(expected_answer)
    MultiVerify(dpr).compute(
        "centrality.betweenness", graph, normalize=False,
    ).assert_equal(expected_answer)


def test_katz_centrality(default_plugin_resolver):
    r"""
              +-+
     ------>  |1| ----------------------------
     |        +-+                            |
     |                                       |
     |         |                             |
     |         v                             |
                                             V
    +-+  <--  +-+       +-+       +-+       +-+
    |0|       |2|  <--  |3|  -->  |4|  <--  |5|
    +-+  -->  +-+       +-+       +-+       +-+
    """
    dpr = default_plugin_resolver
    networkx_graph_data = [
        (0, 1),
        (0, 2),
        (2, 0),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 4),
        (5, 4),
    ]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_edges_from(networkx_graph_data)
    data = {
        0: 0.4069549895218489,
        1: 0.40687482321632046,
        2: 0.41497162410274485,
        3: 0.40280527348222406,
        4: 0.410902066312543,
        5: 0.4068740216338262,
    }
    expected_val = dpr.wrappers.NodeMap.PythonNodeMap(data)
    graph = dpr.wrappers.Graph.NetworkXGraph(networkx_graph)
    MultiVerify(dpr).compute("centrality.katz", graph, tolerance=1e-7).assert_equal(
        expected_val, rel_tol=1e-5
    )


def test_pagerank_centrality(default_plugin_resolver):
    r"""
              +-+
     ------>  |1|
     |        +-+
     |
     |         |
     |         v

    +-+  <--  +-+       +-+
    |0|       |2|  <--  |3|
    +-+  -->  +-+       +-+
    """
    dpr = default_plugin_resolver
    networkx_graph_data = [(0, 1), (0, 2), (2, 0), (1, 2), (3, 2)]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_edges_from(networkx_graph_data)
    data = {
        0: 0.37252685132844066,
        1: 0.19582391181458728,
        2: 0.3941492368569718,
        3: 0.037500000000000006,
    }
    expected_val = dpr.wrappers.NodeMap.PythonNodeMap(data)
    graph = dpr.wrappers.Graph.NetworkXGraph(networkx_graph)
    MultiVerify(dpr).compute(
        dpr.algos.centrality.pagerank, graph, tolerance=1e-7
    ).assert_equal(expected_val, rel_tol=1e-5)


def test_closeness_centrality(default_plugin_resolver):
    dpr = default_plugin_resolver
    graph = build_standard_graph()
    nodes = dpr.wrappers.NodeSet.PythonNodeSet({0, 1, 2, 3, 4, 5, 6, 7})
    expected = dpr.wrappers.NodeMap.PythonNodeMap(
        {
            0: 0.051948051948051945,
            1: 0.03809523809523809,
            2: 0.02990033222591362,
            3: 0.14285714285714285,
            4: 0.08035714285714285,
            5: 0.06679035250463822,
            6: 0.04250295159386069,
            7: 0.03271028037383177,
        }
    )
    MultiVerify(dpr).compute("centrality.closeness", graph).assert_equal(expected)
    MultiVerify(dpr).compute("centrality.closeness", graph, nodes).assert_equal(
        expected
    )


def test_eigenvector_centrality(default_plugin_resolver):
    dpr = default_plugin_resolver
    graph = build_standard_graph()
    expected = dpr.wrappers.NodeMap.PythonNodeMap(
        {
            0: 3.718912841322492e-24,
            1: 4.4815545088477956e-24,
            2: 0.5668908376472616,
            3: 1.54302627577451e-24,
            4: 0.2304676227496987,
            5: 0.4988989293616064,
            6: 0.5070599863741589,
            7: 0.34570143412454807,
        }
    )
    MultiVerify(dpr).compute("centrality.eigenvector", graph).assert_equal(
        expected, rel_tol=1e-3
    )


def test_hits_centrality(default_plugin_resolver):
    dpr = default_plugin_resolver
    graph = build_standard_graph()
    hubs = dpr.wrappers.NodeMap.PythonNodeMap(
        {
            0: 1.0693502568464412e-135,
            1: 0.0940640958864079,
            2: 0.3219827031019462,
            3: 0.36559982252958123,
            4: 0.2183519269850825,
            5: 1.069350256846441e-11,
            6: 1.451486288792823e-06,
            7: 0.0,
        }
    )
    authority = dpr.wrappers.NodeMap.PythonNodeMap(
        {
            0: 0.014756025909040777,
            1: 0.2007333553742929,
            2: 1.5251309332182024e-06,
            3: 1.2359669426636484e-134,
            4: 0.35256375000871987,
            5: 0.2804151003457033,
            6: 1.2359669426636479e-11,
            7: 0.15153024321895017,
        }
    )
    MultiVerify(dpr).compute("centrality.hits", graph, tol=1e-06).assert_equal(
        (hubs, authority), rel_tol=1e-3
    )
