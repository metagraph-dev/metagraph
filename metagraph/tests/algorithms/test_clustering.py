from metagraph.tests.util import default_plugin_resolver
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
        assert len(x) == 8, f"{len(x)} != 8"
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
        assert len(x) == 4, f"{len(x)} != 4"
        c1 = set(x[i] for i in (0, 1, 2))
        c2 = x[3]
        assert len(c1) == 1, c1
        c1 = c1.pop()
        assert c1 != c2, f"{c1}, {c2}"

    MultiVerify(dpr).compute(
        "clustering.strongly_connected_components", graph
    ).normalize(dpr.types.NodeMap.PythonNodeMapType).custom_compare(cmp_func)


def test_triangle_count(default_plugin_resolver):
    dpr = default_plugin_resolver
    # Simple graph with 5 triangles
    # 0 - 1    5 - 6
    # | X |    | /
    # 3 - 4 -- 2 - 7
    simple_graph_data = [
        [0, 1, 100],
        [0, 3, 200],
        [0, 4, 300],
        [1, 3, 50],
        [1, 4, 55],
        [2, 4, 60],
        [2, 5, 65],
        [2, 6, 70],
        [3, 4, 75],
        [5, 6, 20],
        [6, 7, 10],
    ]
    # Build simple graph with 5 triangles
    simple_graph = nx.Graph()
    simple_graph.add_weighted_edges_from(simple_graph_data)
    # Convert to wrapper
    graph = dpr.wrappers.Graph.NetworkXGraph(simple_graph)

    MultiVerify(dpr).compute("cluster.triangle_count", graph).assert_equal(5)


def test_louvain_step(default_plugin_resolver):
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
        assert len(x_graph) == 8, f"{len(x_graph)} != 8"
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
        assert len(x) == 8, f"{len(x)} != 8"
        c1 = set(x[i] for i in (0, 1, 3, 4))
        c2 = set(x[i] for i in (2, 5, 6, 7))
        assert len(c1) == 1, c1
        assert len(c2) == 1, c2
        c1 = c1.pop()
        c2 = c2.pop()
        assert c1 != c2, f"{c1}, {c2}"

    MultiVerify(dpr).compute("clustering.label_propagation_community", graph).normalize(
        dpr.types.NodeMap.PythonNodeMapType
    ).custom_compare(cmp_func)


def test_greedy_coloring(default_plugin_resolver):
    #   0 1 2 3 4 5    Node Coloring
    # 0 - 1 - 1 - 1    0    0
    # 1 1 - 1 - - -    1    1
    # 2 - 1 - - 1 -    2    0
    # 3 1 - - - - 1    3    1
    # 4 - - 1 - - 1    4    1
    # 5 1 - - 1 1 -    5    2
    dpr = default_plugin_resolver
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 3), (0, 5), (1, 2), (2, 4), (3, 5), (4, 5)])
    graph = dpr.wrappers.Graph.NetworkXGraph(g)

    def cmp_func(colors):
        # Check that the triangle in the graph (0, 3, 5) all have different colors
        assert {colors[0], colors[3], colors[5]} == {0, 1, 2}

    results = MultiVerify(dpr).compute("clustering.coloring.greedy", graph)
    # Check number of colors required
    results[1].assert_equal(3)
    # Check coloring of triangle in the graph
    results[0].normalize(dpr.types.NodeMap.PythonNodeMapType).custom_compare(cmp_func)
