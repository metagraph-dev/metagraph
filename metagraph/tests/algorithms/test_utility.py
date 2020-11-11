import pytest
from metagraph.tests.util import default_plugin_resolver
import networkx as nx
import numpy as np
import scipy.sparse as ss
from . import MultiVerify


def test_nodeset_choose_random(default_plugin_resolver):
    dpr = default_plugin_resolver
    py_node_set = {100, 200, 300, 400, 500, 600, 700}
    k = 3

    def cmp_func(x):
        assert len(x) == k
        assert len(x) < len(py_node_set)
        assert x.issubset(py_node_set)

    MultiVerify(dpr).compute("util.nodeset.choose_random", py_node_set, k).normalize(
        dpr.types.NodeSet.PythonNodeSetType
    ).custom_compare(cmp_func)


def test_nodeset_from_vector(default_plugin_resolver):
    dpr = default_plugin_resolver
    np_node_vector = np.array([1, 2, 3])
    MultiVerify(dpr).compute("util.nodeset.from_vector", np_node_vector).assert_equal(
        dpr.wrappers.NodeSet.NumpyNodeSet(np.array([1, 2, 3]))
    )


def test_nodemap_sort(default_plugin_resolver):
    dpr = default_plugin_resolver
    py_node_map = {index: index * 100 for index in range(1, 8)}
    mv = MultiVerify(dpr)
    mv.compute("util.nodemap.sort", py_node_map).assert_equal(
        np.array([1, 2, 3, 4, 5, 6, 7])
    )
    mv.compute("util.nodemap.sort", py_node_map, True, 4).assert_equal(
        np.array([1, 2, 3, 4])
    )
    mv.compute("util.nodemap.sort", py_node_map, True).assert_equal(
        np.array([1, 2, 3, 4, 5, 6, 7])
    )
    mv.compute("util.nodemap.sort", py_node_map, False, 3).assert_equal(
        np.array([7, 6, 5])
    )
    mv.compute("util.nodemap.sort", py_node_map, False).assert_equal(
        np.array([7, 6, 5, 4, 3, 2, 1])
    )


def test_nodemap_select(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = {1: 11, 2: 22, 3: 33, 4: 44}
    node_set = dpr.wrappers.NodeSet.NumpyNodeSet({2, 3})
    correct_answer = {2: 22, 3: 33}
    MultiVerify(dpr).compute("util.nodemap.select", node_map, node_set).assert_equal(
        correct_answer
    )


def test_nodemap_filter(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = {1: 11, 2: 22, 3: 33, 4: 44}
    correct_answer = {2, 4}

    def filter_func(x):
        return x % 2 == 0

    MultiVerify(dpr).compute("util.nodemap.filter", node_map, filter_func).assert_equal(
        correct_answer
    )
    filter_func = lambda x: x % 2 == 0
    MultiVerify(dpr).compute("util.nodemap.filter", node_map, filter_func).assert_equal(
        correct_answer
    )


def test_nodemap_apply(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = {1: 11, 2: 22, 3: 33, 4: 44}
    apply_func = lambda x: x * 100
    correct_answer = {1: 1100, 2: 2200, 3: 3300, 4: 4400}
    MultiVerify(dpr).compute("util.nodemap.apply", node_map, apply_func).assert_equal(
        correct_answer
    )


def test_nodemap_reduce(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = {1: 11, 2: 22, 3: 33, 4: 44}
    reduce_func = lambda x, y: x + y
    correct_answer = 110
    MultiVerify(dpr).compute("util.nodemap.reduce", node_map, reduce_func).assert_equal(
        correct_answer
    )


def test_graph_degree(default_plugin_resolver):
    dpr = default_plugin_resolver
    #   0 1 2 3 4
    # 0 - 5 - - 2
    # 1 - - 2 - -
    # 2 1 - 4 - -
    # 3 1 - - - 7
    # 4 6 - - - -
    g = nx.DiGraph()
    g.add_weighted_edges_from(
        [
            (0, 1, 5),
            (0, 4, 2),
            (1, 2, 2),
            (2, 0, 1),
            (2, 2, 4),
            (3, 0, 1),
            (3, 4, 7),
            (4, 0, 6),
        ]
    )
    graph = dpr.wrappers.Graph.NetworkXGraph(g)
    e_out = {0: 2, 1: 1, 2: 2, 3: 2, 4: 1}
    e_in = {0: 3, 1: 1, 2: 2, 3: 0, 4: 2}
    e_all = {0: 5, 1: 2, 2: 4, 3: 2, 4: 3}
    e_none = {i: 0 for i in range(5)}
    mv = MultiVerify(dpr)
    mv.compute("util.graph.degree", graph).assert_equal(e_out)
    mv.compute("util.graph.degree", graph, in_edges=True, out_edges=False).assert_equal(
        e_in
    )
    mv.compute("util.graph.degree", graph, in_edges=True, out_edges=True).assert_equal(
        e_all
    )
    mv.compute(
        "util.graph.degree", graph, in_edges=False, out_edges=False
    ).assert_equal(e_none)


def test_graph_aggregate_edges_directed(default_plugin_resolver):
    r"""
    0 <--2-- 1        5 --10-> 6
    |      ^ |      ^ ^      /
    |     /  |     /  |     /
    1    7   3    9   5   11
    |   /    |  /     |   /
    v        v /        v
    3 --8--> 4 <--4-- 2 --6--> 7
    """
    import operator

    dpr = default_plugin_resolver
    mv = MultiVerify(dpr)
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
    func = operator.mul
    initial_value = 1.0

    expected_answer = {
        0: 2.0,
        1: 42.0,
        2: 1320.0,
        3: 56.0,
        4: 864.0,
        5: 450.0,
        6: 110.0,
        7: 6.0,
    }
    mv.compute(
        "util.graph.aggregate_edges", graph, func, initial_value, True, True
    ).assert_equal(expected_answer)

    expected_answer = {
        0: 1.0,
        1: 6.0,
        2: 120.0,
        3: 56.0,
        4: 9.0,
        5: 10.0,
        6: 11.0,
        7: 1.0,
    }
    mv.compute(
        "util.graph.aggregate_edges", graph, func, initial_value, False, True
    ).assert_equal(expected_answer)

    expected_answer = {
        0: 2.0,
        1: 7.0,
        2: 11.0,
        3: 1.0,
        4: 96.0,
        5: 45.0,
        6: 10.0,
        7: 6.0,
    }
    mv.compute(
        "util.graph.aggregate_edges", graph, func, initial_value, True, False
    ).assert_equal(expected_answer)

    expected_answer = {node: 1.0 for node in range(8)}
    mv.compute(
        "util.graph.aggregate_edges", graph, func, initial_value, False, False
    ).assert_equal(expected_answer)


def test_graph_aggregate_edges_undirected(default_plugin_resolver):
    r"""
    0 ---2-- 1        5 --10-- 6
    |      / |      / |      /
    |     /  |     /  |     /
    1    7   3    9   5   11
    |  _/    |  /     |  _/
    | /      | /      | /
    3 --8--- 4 ---4-- 2 --6--- 7
    """
    import operator

    dpr = default_plugin_resolver
    mv = MultiVerify(dpr)
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
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
    func = operator.mul
    initial_value = 1.0

    expected_answer = {
        0: 2.0,
        1: 42.0,
        2: 1320.0,
        3: 56.0,
        4: 864.0,
        5: 450.0,
        6: 110.0,
        7: 6.0,
    }
    mv.compute(
        "util.graph.aggregate_edges", graph, func, initial_value, True, True
    ).assert_equal(expected_answer)
    mv.compute(
        "util.graph.aggregate_edges", graph, func, initial_value, False, True
    ).assert_equal(expected_answer)
    mv.compute(
        "util.graph.aggregate_edges", graph, func, initial_value, True, False
    ).assert_equal(expected_answer)

    expected_answer = {node: 1.0 for node in range(8)}
    mv.compute(
        "util.graph.aggregate_edges", graph, func, initial_value, False, False
    ).assert_equal(expected_answer)


def test_graph_filter_edges(default_plugin_resolver):
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
    func = lambda weight: weight > 5

    expected_answer_nx_graph = nx.DiGraph()
    expected_answer_nx_graph.add_nodes_from(range(8))
    expected_answer_nx_graph.add_weighted_edges_from(
        [(2, 7, 6), (3, 1, 7), (3, 4, 8), (4, 5, 9), (5, 6, 10), (6, 2, 11),]
    )
    expected_answer = dpr.wrappers.Graph.NetworkXGraph(expected_answer_nx_graph)
    MultiVerify(dpr).compute("util.graph.filter_edges", graph, func).assert_equal(
        expected_answer
    )


def test_assign_uniform_weight(default_plugin_resolver):
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
        (0, 3),
        (1, 0),
        (1, 4),
        (2, 4),
        (2, 5),
        (2, 7),
        (3, 1),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 2),
    ]
    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from(ebunch)
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
    initial_weight = 100

    expected_answer_nx_graph = nx.DiGraph()
    expected_answer_nx_graph.add_nodes_from(range(8))
    expected_answer_nx_graph.add_weighted_edges_from(
        [
            (0, 3, 100),
            (1, 0, 100),
            (1, 4, 100),
            (2, 4, 100),
            (2, 5, 100),
            (2, 7, 100),
            (3, 1, 100),
            (3, 4, 100),
            (4, 5, 100),
            (5, 6, 100),
            (6, 2, 100),
        ]
    )
    expected_answer = dpr.wrappers.Graph.NetworkXGraph(expected_answer_nx_graph)
    MultiVerify(dpr).compute(
        "util.graph.assign_uniform_weight", graph, initial_weight
    ).assert_equal(expected_answer)


def test_graph_build(default_plugin_resolver):
    dpr = default_plugin_resolver
    mv = MultiVerify(dpr)
    # Edge Map + Node Set
    r"""
    1 --1--- 5      2
    |     _/ |
    |   _9   |
    3  /     2
    | /      |
    3 --4--- 4      9
    """
    edgemap_ss_matrix = ss.csr_matrix(
        np.array(
            [[0, 3, 0, 1], [3, 0, 4, 9], [0, 4, 0, 2], [1, 9, 2, 0]], dtype=np.int64
        )
    )
    graph_ss_matrix = edgemap_ss_matrix.copy()
    graph_ss_matrix.resize(6, 6)
    edgemap = dpr.wrappers.EdgeMap.ScipyEdgeMap(edgemap_ss_matrix, [1, 3, 4, 5])
    nodeset = dpr.wrappers.NodeSet.NumpyNodeSet({1, 2, 3, 4, 5, 9})
    expected_answer = dpr.wrappers.Graph.ScipyGraph(graph_ss_matrix, [1, 3, 4, 5, 2, 9])
    mv.compute("util.graph.build", edgemap, nodeset).assert_equal(expected_answer)

    # Edge Map + Node Map
    r"""
    1(10) --1--- 5(50)      2(98)
    |           /  |
    |    ___9__/   |
    3   /          2
    |  /           |
    3(30) --4--- 4(40)      9(99)
    """
    node_map_data = np.array([99, 10, 98, 30, 40, 50])
    nodemap = dpr.wrappers.NodeMap.NumpyNodeMap(node_map_data, nodes=[9, 1, 2, 3, 4, 5])
    expected_answer = dpr.wrappers.Graph.ScipyGraph(
        graph_ss_matrix, [1, 3, 4, 5, 9, 2], [10, 30, 40, 50, 99, 98]
    )
    mv.compute("util.graph.build", edgemap, nodemap).assert_equal(expected_answer)

    # Edge Set + Node Map
    r"""
    1(10) ------ 5(50)      2(98)
    |           /  |
    |    ______/   |
    |   /          |
    |  /           |
    3(30) ------ 4(40)      9(99)
    """
    edgeset_ss_matrix = ss.csr_matrix(
        np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]], dtype=bool)
    )
    graph_ss_matrix = edgeset_ss_matrix.copy()
    graph_ss_matrix.resize(6, 6)
    edgeset = dpr.wrappers.EdgeSet.ScipyEdgeSet(edgeset_ss_matrix, [1, 3, 4, 5])
    node_map_data = np.array([99, 10, 98, 30, 40, 50])
    nodemap = dpr.wrappers.NodeMap.NumpyNodeMap(node_map_data, nodes=[9, 1, 2, 3, 4, 5])
    expected_answer = dpr.wrappers.Graph.ScipyGraph(
        graph_ss_matrix, [1, 3, 4, 5, 2, 9], [10, 30, 40, 50, 98, 99]
    )
    mv.compute("util.graph.build", edgeset, nodemap).assert_equal(expected_answer)

    # Edge Set + Node Set
    r"""
    1 ------ 5      2
    |     _/ |
    |   _/   |
    |  /     |
    | /      |
    3 ------ 4      9
    """
    nodeset = dpr.wrappers.NodeSet.NumpyNodeSet(np.array([9, 1, 2, 3, 4, 5]))
    expected_answer = dpr.wrappers.Graph.ScipyGraph(graph_ss_matrix, [1, 3, 4, 5, 9, 2])
    mv.compute("util.graph.build", edgeset, nodeset).assert_equal(expected_answer)


def test_edgemap_from_edgeset(default_plugin_resolver):
    dpr = default_plugin_resolver
    #    0 2 7
    # 0 [    1]
    # 2 [    1]
    # 7 [1 1  ]
    matrix = ss.coo_matrix(([1, 1, 1, 1], ([0, 1, 2, 2], [2, 2, 0, 1])), dtype=np.int64)
    edgeset = dpr.wrappers.EdgeSet.ScipyEdgeSet(matrix, [0, 2, 7])
    #    0 2 7
    # 0 [    9]
    # 2 [    9]
    # 7 [9 9  ]
    expected_matrix = ss.coo_matrix(
        ([9, 9, 9, 9], ([0, 1, 2, 2], [2, 2, 0, 1])), dtype=np.int64
    )
    expected_answer = dpr.wrappers.EdgeMap.ScipyEdgeMap(expected_matrix, [0, 2, 7])
    MultiVerify(dpr).compute("util.edgemap.from_edgeset", edgeset, 9).assert_equal(
        expected_answer
    )


def test_node_embedding_apply(default_plugin_resolver):
    dpr = default_plugin_resolver
    matrix = np.arange(6).reshape(2, 3)
    node_map = dpr.wrappers.NodeMap.NumpyNodeMap(
        np.array([0, 1]), nodes=np.array([9990, 9991])
    )

    MultiVerify(dpr).compute(
        "util.node_embedding.apply", matrix, node_map, np.array([9990])
    ).assert_equal(np.array([[0, 1, 2]]))
    MultiVerify(dpr).compute(
        "util.node_embedding.apply", matrix, node_map, np.array([9991])
    ).assert_equal(np.array([[3, 4, 5]]))
    MultiVerify(dpr).compute(
        "util.node_embedding.apply", matrix, node_map, np.array([9990, 9991])
    ).assert_equal(matrix)
    MultiVerify(dpr).compute(
        "util.node_embedding.apply", matrix, node_map, np.array([9991, 9990])
    ).assert_equal(np.array([[3, 4, 5], [0, 1, 2]]))


def test_isomorphic(default_plugin_resolver):
    dpr = default_plugin_resolver
    #   0 1 2 3 4            0 1 2 3 4
    # 0 1 1 - 1 -     2 -> 0 - - - 1 1
    # 1 - - 1 - -     4 -> 1 - - 1 - -
    # 2 1 1 - - -     3 -> 2 1 - 1 - 1
    # 3 - 1 1 - -     0 -> 3 - - 1 - 1
    # 4 1 - - - -     1 -> 4 - - - 1 -
    g1 = nx.DiGraph()
    g1.add_edges_from(
        [(0, 0), (0, 1), (0, 3), (1, 2), (2, 0), (2, 1), (3, 1), (3, 2), (4, 0)]
    )
    g2 = nx.DiGraph()
    g2.add_edges_from(
        [(0, 3), (0, 4), (1, 2), (2, 0), (2, 2), (2, 4), (3, 2), (3, 4), (4, 3)]
    )
    graph1 = dpr.wrappers.Graph.NetworkXGraph(g1)
    graph2 = dpr.wrappers.Graph.NetworkXGraph(g2)
    MultiVerify(dpr).compute("util.graph.isomorphic", graph1, graph2).assert_equal(True)
