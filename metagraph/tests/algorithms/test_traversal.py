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
    # TODO test depth_limit when supported in SciPy
    MultiVerify(dpr).compute("traversal.bfs_iter", graph, 0).normalize(
        dpr.types.Vector.NumpyVectorType
    ).assert_equal(correct_answer)


def test_bfs_tree(default_plugin_resolver):
    r"""
    0 --2--> 1        5 --10-> 6
    |      _/|      ^ ^      /
    |     /  |     /  |     /
    1   12   3    9   5   11
    |  /     |  /     |   /
    v v      v /        v
    3 --8--> 4 <--4-- 2 --6--> 7
    """

    dpr = default_plugin_resolver
    ebunch = [
        (0, 3, 1),
        (0, 1, 2),
        (1, 3, 12),
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

    # No Depth Limit

    expected_node2depth = {
        0: 0,
        1: 1,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        2: 5,
        7: 6,
    }

    expected_node2parent_candidates = {
        0: [0],
        1: [0],
        3: [0],
        4: [1, 3],
        5: [4],
        6: [5],
        2: [6],
        7: [2],
    }

    def cmp_func(answer):
        node2depth, node2parent = answer
        assert node2depth == expected_node2depth
        assert len(node2parent) == len(expected_node2parent_candidates)
        for node, parent_candidates in expected_node2parent_candidates.items():
            assert node2parent[node] in parent_candidates

    MultiVerify(dpr).compute("traversal.bfs_tree", graph, 0).normalize(
        (dpr.types.NodeMap.PythonNodeMapType, dpr.types.NodeMap.PythonNodeMapType)
    ).custom_compare(cmp_func)

    # Depth Limit

    expected_node2depth = {
        0: 0,
        1: 1,
        3: 1,
        4: 2,
        5: 3,
    }

    expected_node2parent_candidates = {
        0: [0],
        1: [0],
        3: [0],
        4: [1, 3],
        5: [4],
    }

    def cmp_func(answer):
        node2depth, node2parent = answer
        assert node2depth == expected_node2depth
        assert len(node2parent) == len(expected_node2parent_candidates)
        for node, parent_candidates in expected_node2parent_candidates.items():
            assert node2parent[node] in parent_candidates

    MultiVerify(dpr).compute("traversal.bfs_tree", graph, 0, 3).normalize(
        (dpr.types.NodeMap.PythonNodeMapType, dpr.types.NodeMap.PythonNodeMapType)
    ).custom_compare(cmp_func)


def test_dfs_iter(default_plugin_resolver):
    r"""
    0 --2--> 1        5 --10-> 6
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
        (0, 1, 2),
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

    def cmp_func(dfs_vector):
        node_one_position = np.flatnonzero(dfs_vector == 1).item()
        node_three_position = np.flatnonzero(dfs_vector == 3).item()
        assert abs(node_one_position - node_three_position) == 6
        assert dfs_vector[0] == 0
        assert np.all(dfs_vector[2:7] == np.array([4, 5, 6, 2, 7]))

    MultiVerify(dpr).compute("traversal.dfs_iter", graph, 0).normalize(
        dpr.types.Vector.NumpyVectorType
    ).custom_compare(cmp_func)


def test_dfs_tree(default_plugin_resolver):
    r"""
    0 --2--> 1        5 --10-> 6
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
        (0, 1, 2),
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

    expected_results_strict = {
        0: 0,
        5: 4,
        6: 5,
        2: 6,
        7: 2,
    }

    def cmp_func(node2parent):
        assert len(node2parent) == 8
        for node, expected_parent in expected_results_strict.items():
            assert node2parent[node] == expected_parent
        assert node2parent[1] in [0, 7]
        assert node2parent[3] in [0, 7]
        assert node2parent[4] in [1, 3]

    MultiVerify(dpr).compute("traversal.dfs_tree", graph, 0).normalize(
        dpr.types.NodeMap.PythonNodeMapType
    ).custom_compare(cmp_func)


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


def test_astar(default_plugin_resolver):
    r"""
    00 - 01 - 02 - 03 - 04 - 05 - 06 - 07 - 08 - 09
    |  x  | x  | x  | x  | x  | x  | x  | x  | x  |
    10 - 11 - 12 - 13 - 14 - 15 - 16 - 17 - 18 - 19
    |  x  | x  | x  | x  | x  | x  | x  | x  | x  |
    20 - 21 - 22 - 23 - 24 - 25 - 26 - 27 - 28 - 29
    |  x  | x  | x  | x  | x  | x  | x  | x  | x  |
    30 - 31 - 32 - 33 - 34 - 35 - 36 - 37 - 38 - 39
    | ____|                                  |___ |
    40____                                   ____49
    |     |                                  |    |
    50 - 51 - 52 - 53 - 54 - 55 - 56 - 57 - 58 - 59
    |  x  | x  | x  | x  | x  | x  | x  | x  | x  |
    60 - 61 - 62 - 63 - 64 - 65 - 66 - 67 - 68 - 69
    |  x  | x  | x  | x  | x  | x  | x  | x  | x  |
    70 - 71 - 72 - 73 - 74 - 75 - 76 - 77 - 78 - 79
    |  x  | x  | x  | x  | x  | x  | x  | x  | x  |
    80 - 81 - 82 - 83 - 84 - 85 - 86 - 87 - 88 - 89
    |  x  | x  | x  | x  | x  | x  | x  | x  | x  |
    90 - 91 - 92 - 93 - 94 - 95 - 96 - 97 - 98 - 99
    """
    dpr = default_plugin_resolver
    nx_graph = nx.DiGraph()
    node2id = lambda node: node[0] * 10 + node[1]
    excluded_nodes = {
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),
    }
    nodes = ((x, y) for x in range(10) for y in range(10))
    nodes = {node for node in nodes if node not in excluded_nodes}
    for node in nodes:
        neighbors = (
            (node[0] + dx, node[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        )
        neighbors = filter(lambda neighbor: neighbor != node, neighbors)
        neighbors = filter(lambda neighbor: neighbor in nodes, neighbors)
        for neighbor in neighbors:
            nx_graph.add_edge(node2id(node), node2id(neighbor))
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)

    def distance_func(src_id):
        src_x, src_y = tuple(map(int, f"{src_id:02}"))
        dist = (9 - src_x) ** 2 + (9 - src_y) ** 2
        return dist

    expected_path = np.array(
        [00, 11, 22, 33, 34, 35, 36, 37, 38, 49, 59, 69, 79, 89, 99], dtype=int
    )
    MultiVerify(dpr).compute(
        "traversal.astar_search", graph, 00, 99, distance_func
    ).assert_equal(expected_path)
