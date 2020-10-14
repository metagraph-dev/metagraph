import pytest
from metagraph.tests.util import default_plugin_resolver
import networkx as nx
from . import MultiVerify


def test_extract_graph(default_plugin_resolver):
    r"""
    0 ---2-- 1        5 --10-- 6
           / |        |      /
          /  |        |     /
        7    3        5   11
       /     |        |  /
      /      |        | /
    3        4        2 --6--- 7
    """
    dpr = default_plugin_resolver
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(
        [(1, 0, 2), (1, 4, 3), (2, 5, 5), (2, 7, 6), (3, 1, 7), (5, 6, 10), (6, 2, 11),]
    )
    desired_nodes = {2, 5, 6}
    nx_extracted_graph = nx.Graph()
    nx_extracted_graph.add_weighted_edges_from([(2, 5, 5), (5, 6, 10), (6, 2, 11)])
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
    desired_nodes_wrapped = dpr.wrappers.NodeSet.PythonNodeSet(desired_nodes)
    extracted_graph = dpr.wrappers.Graph.NetworkXGraph(nx_extracted_graph)
    MultiVerify(dpr).compute(
        "subgraph.extract_subgraph", graph, desired_nodes_wrapped
    ).assert_equal(extracted_graph)


def test_k_core(default_plugin_resolver):
    r"""
    0 ---2-- 1        5 --10-- 6
           / |        |      /
          /  |        |     /
        7    3        5   11
       /     |        |  /
      /      |        | /
    3        4        2 --6--- 7
    """
    dpr = default_plugin_resolver
    k = 2
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(
        [(1, 0, 2), (1, 4, 3), (2, 5, 5), (2, 7, 6), (3, 1, 7), (5, 6, 10), (6, 2, 11),]
    )
    nx_k_core_graph = nx.Graph()
    nx_k_core_graph.add_weighted_edges_from(
        [(2, 5, 5), (5, 6, 10), (6, 2, 11),]
    )
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
    k_core_graph = dpr.wrappers.Graph.NetworkXGraph(nx_k_core_graph)
    MultiVerify(dpr).compute("subgraph.k_core", graph, k).assert_equal(k_core_graph)


def test_k_truss(default_plugin_resolver):
    r"""
    0 ---- 1 ---- 2
    |\    /|    / |
    | \  / |   /  |
    |  \/  |  /   |
    |  /\  | /    |
    | /  \ |/     |
    3 -----4      5
    """
    dpr = default_plugin_resolver
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(
        [(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (3, 4)]
    )
    nx_3_truss_graph = nx.Graph(
        [(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
    )
    nx_4_truss_graph = nx.Graph([(0, 1), (0, 3), (0, 4), (1, 3), (1, 4), (3, 4)])
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
    three_truss_graph = dpr.wrappers.Graph.NetworkXGraph(nx_3_truss_graph)
    four_truss_graph = dpr.wrappers.Graph.NetworkXGraph(nx_4_truss_graph)
    MultiVerify(dpr).compute("subgraph.k_truss", graph, 3).assert_equal(
        three_truss_graph
    )
    MultiVerify(dpr).compute("subgraph.k_truss", graph, 4).assert_equal(
        four_truss_graph
    )


def test_maximial_independent_set(default_plugin_resolver):
    dpr = default_plugin_resolver
    g = nx.generators.classic.barbell_graph(5, 6)
    graph = dpr.wrappers.Graph.NetworkXGraph(g)

    def cmp_func(nodeset):
        # Verify that every node in the graph is either:
        # 1. in the nodeset
        # 2. directly connected to the nodeset
        ns = nodeset.value
        for node in g.nodes():
            if node in ns:
                continue
            for nbr in g.neighbors(node):
                if nbr in ns:
                    break
            else:
                raise AssertionError(f"node {node} is independent of the set")
        # Verify that nodes in the nodeset are not connected to each other
        for node in ns:
            for nbr in g.neighbors(node):
                assert nbr not in ns, f"nodes {node} and {nbr} are connected"

    MultiVerify(dpr).compute("subgraph.maximal_independent_set", graph).normalize(
        dpr.wrappers.NodeSet.PythonNodeSet
    ).custom_compare(cmp_func)


def test_subisomorphic(default_plugin_resolver):
    pytest.xfail()


def test_node_sampling(default_plugin_resolver):
    pytest.xfail()


def test_edge_sampling(default_plugin_resolver):
    pytest.xfail()


def test_totally_induced_edge_sampling(default_plugin_resolver):
    pytest.xfail()


def test_random_walk_sampling(default_plugin_resolver):
    pytest.xfail()
