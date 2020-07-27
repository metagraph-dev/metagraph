from metagraph.tests.util import default_plugin_resolver
import networkx as nx
import numpy as np
from . import MultiVerify


def test_nodeset_choose_random(default_plugin_resolver):
    dpr = default_plugin_resolver
    py_node_set_unwrapped = {100, 200, 300, 400, 500, 600, 700}
    py_node_set = dpr.wrappers.NodeSet.PythonNodeSet(py_node_set_unwrapped)
    k = 3

    def cmp_func(x):
        assert x.num_nodes == k
        assert x.num_nodes < len(py_node_set_unwrapped)
        assert x.value.issubset(py_node_set_unwrapped)

    MultiVerify(dpr, "util.nodeset.choose_random", py_node_set, k).custom_compare(
        cmp_func, dpr.types.NodeSet.PythonNodeSetType
    )


def test_nodeset_sort(default_plugin_resolver):
    dpr = default_plugin_resolver
    py_node_map_unwrapped = {index: index * 100 for index in range(1, 8)}
    py_node_map = dpr.wrappers.NodeMap.PythonNodeMap(py_node_map_unwrapped)
    MultiVerify(dpr, "util.nodemap.sort", py_node_map).assert_equals(
        dpr.wrappers.Vector.NumpyVector(np.array([1, 2, 3, 4, 5, 6, 7]))
    )
    MultiVerify(dpr, "util.nodemap.sort", py_node_map, True, 4).assert_equals(
        dpr.wrappers.Vector.NumpyVector(np.array([1, 2, 3, 4]))
    )
    MultiVerify(dpr, "util.nodemap.sort", py_node_map, True).assert_equals(
        dpr.wrappers.Vector.NumpyVector(np.array([1, 2, 3, 4, 5, 6, 7]))
    )
    MultiVerify(dpr, "util.nodemap.sort", py_node_map, False, 3).assert_equals(
        dpr.wrappers.Vector.NumpyVector(np.array([7, 6, 5]))
    )
    MultiVerify(dpr, "util.nodemap.sort", py_node_map, False).assert_equals(
        dpr.wrappers.Vector.NumpyVector(np.array([7, 6, 5, 4, 3, 2, 1]))
    )


def test_nodemap_select(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = dpr.wrappers.NodeMap.PythonNodeMap({1: 11, 2: 22, 3: 33, 4: 44})
    node_set = dpr.wrappers.NodeSet.PythonNodeSet({2, 3})
    correct_answer = dpr.wrappers.NodeMap.PythonNodeMap({2: 22, 3: 33})
    MultiVerify(dpr, "util.nodemap.select", node_map, node_set).assert_equals(
        correct_answer
    )


def test_nodemap_filter(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = dpr.wrappers.NodeMap.PythonNodeMap({1: 11, 2: 22, 3: 33, 4: 44})
    correct_answer = dpr.wrappers.NodeSet.PythonNodeSet({2, 4})

    def filter_func(x):
        return x % 2 == 0

    MultiVerify(dpr, "util.nodemap.filter", node_map, filter_func).assert_equals(
        correct_answer
    )
    filter_func = lambda x: x % 2 == 0
    MultiVerify(dpr, "util.nodemap.filter", node_map, filter_func).assert_equals(
        correct_answer
    )


def test_nodemap_apply(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = dpr.wrappers.NodeMap.PythonNodeMap({1: 11, 2: 22, 3: 33, 4: 44})
    apply_func = lambda x: x * 100
    correct_answer = dpr.wrappers.NodeMap.PythonNodeMap(
        {1: 1100, 2: 2200, 3: 3300, 4: 4400}
    )
    MultiVerify(dpr, "util.nodemap.apply", node_map, apply_func).assert_equals(
        correct_answer
    )


def test_nodemap_reduce(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = dpr.wrappers.NodeMap.PythonNodeMap({1: 11, 2: 22, 3: 33, 4: 44})
    reduce_func = lambda x, y: x + y
    correct_answer = 110
    MultiVerify(dpr, "util.nodemap.reduce", node_map, reduce_func).assert_equals(
        correct_answer
    )


def test_graph_aggregate_edges_directed(default_plugin_resolver):
    """
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

    expected_answer = dpr.wrappers.NodeMap.PythonNodeMap(
        {0: 2.0, 1: 42.0, 2: 1320.0, 3: 56.0, 4: 864.0, 5: 450.0, 6: 110.0, 7: 6.0,}
    )
    MultiVerify(
        dpr, "util.graph.aggregate_edges", graph, func, initial_value, True, True
    ).assert_equals(expected_answer)

    expected_answer = dpr.wrappers.NodeMap.PythonNodeMap(
        {0: 1.0, 1: 6.0, 2: 120.0, 3: 56.0, 4: 9.0, 5: 10.0, 6: 11.0, 7: 1.0,}
    )
    MultiVerify(
        dpr, "util.graph.aggregate_edges", graph, func, initial_value, False, True
    ).assert_equals(expected_answer)

    expected_answer = dpr.wrappers.NodeMap.PythonNodeMap(
        {0: 2.0, 1: 7.0, 2: 11.0, 3: 1.0, 4: 96.0, 5: 45.0, 6: 10.0, 7: 6.0,}
    )
    MultiVerify(
        dpr, "util.graph.aggregate_edges", graph, func, initial_value, True, False
    ).assert_equals(expected_answer)

    expected_answer = dpr.wrappers.NodeMap.PythonNodeMap(
        {node: 1.0 for node in range(8)}
    )
    MultiVerify(
        dpr, "util.graph.aggregate_edges", graph, func, initial_value, False, False
    ).assert_equals(expected_answer)


def test_graph_aggregate_edges_undirected(default_plugin_resolver):
    """
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

    expected_answer = dpr.wrappers.NodeMap.PythonNodeMap(
        {0: 2.0, 1: 42.0, 2: 1320.0, 3: 56.0, 4: 864.0, 5: 450.0, 6: 110.0, 7: 6.0,}
    )
    MultiVerify(
        dpr, "util.graph.aggregate_edges", graph, func, initial_value, True, True
    ).assert_equals(expected_answer)
    MultiVerify(
        dpr, "util.graph.aggregate_edges", graph, func, initial_value, False, True
    ).assert_equals(expected_answer)
    MultiVerify(
        dpr, "util.graph.aggregate_edges", graph, func, initial_value, True, False
    ).assert_equals(expected_answer)

    expected_answer = dpr.wrappers.NodeMap.PythonNodeMap(
        {node: 1.0 for node in range(8)}
    )
    MultiVerify(
        dpr, "util.graph.aggregate_edges", graph, func, initial_value, False, False
    ).assert_equals(expected_answer)


def test_graph_filter_edges(default_plugin_resolver):
    """
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

    expected_answer_nx_graph = nx.Graph()
    expected_answer_nx_graph.add_nodes_from(range(8))
    expected_answer_nx_graph.add_weighted_edges_from(
        [(2, 7, 6), (3, 1, 7), (3, 4, 8), (4, 5, 9), (5, 6, 10), (6, 2, 11),]
    )
    expected_answer = dpr.wrappers.Graph.NetworkXGraph(expected_answer_nx_graph)
    MultiVerify(dpr, "util.graph.filter_edges", graph, func).assert_equals(
        expected_answer
    )


def test_add_uniform_weight(default_plugin_resolver):
    """
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
    weight_delta = 1000

    expected_answer_nx_graph = nx.DiGraph()
    expected_answer_nx_graph.add_nodes_from(range(8))
    expected_answer_nx_graph.add_weighted_edges_from(
        [
            (0, 3, 1001),
            (1, 0, 1002),
            (1, 4, 1003),
            (2, 4, 1004),
            (2, 5, 1005),
            (2, 7, 1006),
            (3, 1, 1007),
            (3, 4, 1008),
            (4, 5, 1009),
            (5, 6, 1010),
            (6, 2, 1011),
        ]
    )
    expected_answer = dpr.wrappers.Graph.NetworkXGraph(expected_answer_nx_graph)
    MultiVerify(
        dpr, "util.graph.add_uniform_weight", graph, weight_delta
    ).assert_equals(expected_answer)
