from metagraph.tests.util import default_plugin_resolver
import networkx as nx
import numpy as np
import scipy.sparse as ss
from . import MultiVerify


def test_all_pairs_shortest_paths(default_plugin_resolver):
    r"""
    A --1--- B
    |     _/ |
    |   _9   |
    3  /     2
    | /      |
    C --4--- D
    """
    dpr = default_plugin_resolver
    node_list = [12, 13, 4, 19]
    graph_ss_matrix = ss.csr_matrix(
        np.array(
            [[0, 1, 3, 0], [1, 0, 9, 2], [3, 9, 0, 4], [0, 2, 4, 0]], dtype=np.int64
        )
    )
    graph = dpr.wrappers.Graph.ScipyGraph(graph_ss_matrix, node_list)
    parents_ss_matrix = ss.csr_matrix(
        np.array(
            [[0, 0, 0, 1], [1, 0, 0, 1], [2, 0, 0, 2], [1, 3, 3, 0]], dtype=np.int64
        )
    )
    lengths_ss_matrix = ss.csr_matrix(
        np.array(
            [[0, 1, 3, 3], [1, 0, 4, 2], [3, 4, 0, 4], [3, 2, 4, 0]], dtype=np.float64
        )
    )
    expected_answer = (
        dpr.wrappers.Graph.ScipyGraph(parents_ss_matrix, node_list),
        dpr.wrappers.Graph.ScipyGraph(lengths_ss_matrix, node_list),
    )
    MultiVerify(dpr).compute("traversal.all_pairs_shortest_paths", graph).assert_equal(
        expected_answer
    )


def test_bfs_iter(default_plugin_resolver):
    r"""
    0 <--2-- 1        5 --10-> 6
    |        |      ^ ^      /
    |        |     /  |     /
    1        3    9   5   11
    |        |  /     |   /
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
        (3, 4, 8),
        (4, 5, 9),
        (5, 6, 10),
        (6, 2, 11),
    ]
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph, edge_weight_label="weight")
    correct_answer = np.array([0, 3, 4, 5, 6, 2, 7])
    MultiVerify(dpr).compute("traversal.bfs_iter", graph, 0).assert_equal(
        correct_answer
    )


def test_bellman_ford(default_plugin_resolver):
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
    node_to_parent_mapping = {0: 0, 3: 0, 1: 3, 4: 3, 5: 4, 6: 5, 2: 6, 7: 2}
    node_to_length_mapping = {0: 0, 3: 1, 1: 8, 4: 9, 5: 18, 6: 28, 2: 39, 7: 45}
    expected_answer = (node_to_parent_mapping, node_to_length_mapping)
    MultiVerify(dpr).compute("traversal.bellman_ford", graph, 0).assert_equal(
        expected_answer
    )


def test_dijkstra(default_plugin_resolver):
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
    node_to_parent_mapping = {0: 0, 3: 0, 1: 3, 4: 3, 5: 4, 6: 5, 2: 6, 7: 2}
    node_to_length_mapping = {0: 0, 3: 1, 1: 8, 4: 9, 5: 18, 6: 28, 2: 39, 7: 45}
    expected_answer = (node_to_parent_mapping, node_to_length_mapping)
    MultiVerify(dpr).compute("traversal.dijkstra", graph, 0).assert_equal(
        expected_answer
    )


def test_minimum_spanning_tree(default_plugin_resolver):
    r"""
    0 ---2-- 1        5 --10-- 6
    |      / |      / |      /
    |     /  |     /  |     /
    1    7   3    9   5   11
    |  _/    |  /     |  /
    | /      | /      | /
    3 --8--- 4 ---4-- 2 --6--- 7
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
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)

    ebunch_answer = [
        (0, 3, 1),
        (0, 1, 2),
        (1, 4, 3),
        (4, 2, 4),
        (2, 5, 5),
        (2, 7, 6),
        (5, 6, 10),
    ]
    nx_graph_answer = nx.Graph()
    nx_graph_answer.add_weighted_edges_from(ebunch_answer)
    expected_answer = dpr.wrappers.Graph.NetworkXGraph(nx_graph_answer)
    MultiVerify(dpr).compute("traversal.minimum_spanning_tree", graph).assert_equal(
        expected_answer
    )


def test_minimum_spanning_tree_disconnected(default_plugin_resolver):
    r"""
    0 ---2-- 1        5 --10-- 6
    |      / |        |      /
    |     /  |        |     /
    1    7   3        5   11
    |  _/    |        |  /
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

    ebunch_answer = [
        (0, 3, 1),
        (0, 1, 2),
        (1, 4, 3),
        (2, 5, 5),
        (2, 7, 6),
        (5, 6, 10),
    ]
    nx_graph_answer = nx.Graph()
    nx_graph_answer.add_weighted_edges_from(ebunch_answer)
    expected_answer = dpr.wrappers.Graph.NetworkXGraph(nx_graph_answer)
    MultiVerify(dpr).compute("traversal.minimum_spanning_tree", graph).assert_equal(
        expected_answer
    )
