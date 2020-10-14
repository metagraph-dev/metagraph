import pytest
from metagraph.tests.util import default_plugin_resolver
import networkx as nx
import numpy as np
from . import MultiVerify


def test_degree_centrality(default_plugin_resolver):
    pytest.xfail()


def test_betweenness_centrality_single_hub(default_plugin_resolver):
    r"""
    0 <--2-- 1        5 --10-> 6
    |      ^ |      ^ ^      /
    |     /  |     /  |     /
    1    7   3    9   5   11
    |   /    |  /     |   /
    v        v /        v
    3 --8--> 4 <--4-- 2 --6--> 7
    """
    dpr = default_plugin_resolver
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
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
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
    pytest.xfail()


def test_eigenvector_centrality(default_plugin_resolver):
    pytest.xfail()


def test_hits_centrality(default_plugin_resolver):
    pytest.xfail()
