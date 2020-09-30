from metagraph.tests.util import default_plugin_resolver
from metagraph.plugins.python.types import PythonNodeMap
import networkx as nx
from typing import Tuple
from . import MultiVerify


def test_connected_components(default_plugin_resolver):
    r"""
    0 ---2-- 1        5 --10-- 6
    |      / |        |      /
    |     /  |        |     /
    1   7    3        5   11
    |  /     |        |  /
    | /      |        | /
    3 --8--- 4        2 --6--- 7
    """
    dpr = default_plugin_resolver
    ebunch = [
        (0, 3, 1),
        (1, 0, 2),
        (1, 4, 3),
        (2, 5, 5),
        (2, 7, 6),
        (3, 1, 7),
        (3, 4, 8),
        (5, 6, 10),
        (6, 2, 11),
    ]
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph, edge_weight_label="weight")

    def cmp_func(x):
        # clusters should be:
        # [0, 1, 3, 4]
        # [2, 5, 6, 7]
        assert x.num_nodes == 8, x.num_nodes
        c1 = set(x[i] for i in (0, 1, 3, 4))
        c2 = set(x[i] for i in (2, 5, 6, 7))
        assert len(c1) == 1, c1
        assert len(c2) == 1, c2
        c1 = c1.pop()
        c2 = c2.pop()
        assert c1 != c2, f"{c1}, {c2}"

    (
        MultiVerify(dpr)
        .compute("clustering.connected_components", graph)
        .normalize(dpr.types.NodeMap.PythonNodeMapType)
        .custom_compare(cmp_func)
    )


def test_strongly_connected_components(default_plugin_resolver):
    r"""
              +-+
     ----9->  |1|
     |        +-+
     |
     |         |
     |         6
     |         |
     |         v

    +-+  <-7-  +-+        +-+
    |0|        |2|  <-5-  |3|
    +-+  -8->  +-+        +-+
    """
    dpr = default_plugin_resolver
    networkx_graph_data = [(0, 1, 9), (0, 2, 8), (2, 0, 7), (1, 2, 6), (3, 2, 5)]
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(networkx_graph_data, weight="wait")
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph, edge_weight_label="wait")

    def cmp_func(x):
        # clusters should be:
        # [0, 1, 2]
        # [3]
        assert x.num_nodes == 4, x.num_nodes
        c1 = set(x[i] for i in (0, 1, 2))
        c2 = x[3]
        assert len(c1) == 1, c1
        c1 = c1.pop()
        assert c1 != c2, f"{c1}, {c2}"

    MultiVerify(dpr).compute(
        "clustering.strongly_connected_components", graph
    ).normalize(PythonNodeMap.Type).custom_compare(cmp_func)


def test_louvain(default_plugin_resolver):
    r"""
    0 ---2-- 1        5 --10-- 6
    |      / |        |      /
    |     /  |        |     /
    1   7    3        5   11
    |  /     |        |  /
    | /      |        | /
    3 --8--- 4        2 --6--- 7
    """
    dpr = default_plugin_resolver
    ebunch = [
        (0, 3, 1),
        (1, 0, 2),
        (1, 4, 3),
        (2, 5, 5),
        (2, 7, 6),
        (3, 1, 7),
        (3, 4, 8),
        (5, 6, 10),
        (6, 2, 11),
    ]
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)

    def cmp_func(x):
        x_graph, modularity_score = x
        assert x_graph.num_nodes == 8, x_graph.num_nodes
        assert modularity_score > 0.45

    MultiVerify(dpr).compute("clustering.louvain_community", graph).custom_compare(
        cmp_func
    )


def test_label_propagation(default_plugin_resolver):
    r"""
    0 ---2-- 1        5 --10-- 6
    |      / |        |      /
    |     /  |        |     /
    1   7    3        5   11
    |  /     |        |  /
    | /      |        | /
    3 --8--- 4        2 --6--- 7
    """
    dpr = default_plugin_resolver
    ebunch = [
        (0, 3, 1),
        (1, 0, 2),
        (1, 4, 3),
        (2, 5, 5),
        (2, 7, 6),
        (3, 1, 7),
        (3, 4, 8),
        (5, 6, 10),
        (6, 2, 11),
    ]
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)

    def cmp_func(x):
        # clusters should be:
        # [0, 1, 3, 4]
        # [2, 5, 6, 7]
        assert x.num_nodes == 8, x.num_nodes
        c1 = set(x[i] for i in (0, 1, 3, 4))
        c2 = set(x[i] for i in (2, 5, 6, 7))
        assert len(c1) == 1, c1
        assert len(c2) == 1, c2
        c1 = c1.pop()
        c2 = c2.pop()
        assert c1 != c2, f"{c1}, {c2}"

    MultiVerify(dpr).compute("clustering.label_propagation_community", graph).normalize(
        PythonNodeMap
    ).custom_compare(cmp_func)
