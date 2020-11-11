import pytest
from metagraph.tests.util import default_plugin_resolver
import networkx as nx
from . import MultiVerify


def test_extract_subgraph(default_plugin_resolver):
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
    extracted_graph = dpr.wrappers.Graph.NetworkXGraph(nx_extracted_graph)
    MultiVerify(dpr).compute(
        "subgraph.extract_subgraph", graph, desired_nodes
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

    def cmp_func(ns):
        # Verify that every node in the graph is either:
        # 1. in the nodeset
        # 2. directly connected to the nodeset
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
        dpr.types.NodeSet.PythonNodeSetType
    ).custom_compare(cmp_func)


def test_subisomorphic(default_plugin_resolver):
    dpr = default_plugin_resolver
    #   0 1 2 3 4 | 5 6 7 8             0 1 2 3 4
    # 0 1 1 - 1 - | - 1 - -      2 -> 0 - - - 1 1
    # 1 - - 1 - - | - - - -      4 -> 1 - - 1 - -
    # 2 1 1 - - - | 1 - 1 1      3 -> 2 1 - 1 - 1
    # 3 - 1 1 - - | - - - 1      0 -> 3 - - 1 - 1
    # 4 1 - - - - | - 1 - 1      1 -> 4 - - - 1 -
    # -------------
    # 5 - - 1 - 1   - - - -
    # 6 - - - - 1   1 - 1 -
    # 7 - 1 - - 1   - 1 1 -
    # 8 - - 1 - -   1 1 - -
    big_g = nx.DiGraph()
    big_g.add_edges_from(
        [
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 6),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 5),
            (2, 7),
            (2, 8),
            (3, 1),
            (3, 2),
            (3, 8),
            (4, 0),
            (4, 6),
            (4, 8),
            (5, 2),
            (5, 4),
            (6, 4),
            (6, 5),
            (6, 7),
            (7, 1),
            (7, 4),
            (7, 6),
            (7, 7),
            (8, 2),
            (8, 5),
            (8, 6),
        ]
    )
    g1 = nx.DiGraph()
    g1.add_edges_from(
        [(0, 0), (0, 1), (0, 3), (1, 2), (2, 0), (2, 1), (3, 1), (3, 2), (4, 0)]
    )
    g2 = nx.DiGraph()
    g2.add_edges_from(
        [(0, 3), (0, 4), (1, 2), (2, 0), (2, 2), (2, 4), (3, 2), (3, 4), (4, 3)]
    )
    big_graph = dpr.wrappers.Graph.NetworkXGraph(big_g)
    graph1 = dpr.wrappers.Graph.NetworkXGraph(g1)
    graph2 = dpr.wrappers.Graph.NetworkXGraph(g2)
    MultiVerify(dpr).compute("subgraph.subisomorphic", big_graph, graph1).assert_equal(
        True
    )
    MultiVerify(dpr).compute("subgraph.subisomorphic", big_graph, graph2).assert_equal(
        True
    )


def test_node_sampling(default_plugin_resolver):
    dpr = default_plugin_resolver
    # Build a complete graph, then add a bunch of disconnected nodes
    # Node Sampling should pick some of the disconnected nodes for the subgraph
    g = nx.complete_graph(25)
    g.add_nodes_from(range(25, 50))
    graph = dpr.wrappers.Graph.NetworkXGraph(g)

    def cmp_func(subgraph):
        subg = subgraph.value
        assert 0 < len(subg.nodes()) < len(g.nodes()), f"# nodes = {len(subg.nodes())}"
        # Verify some of the isolated nodes were chosen
        assert subg.nodes() & set(range(25, 50)), f"no isolated nodes found in subgraph"
        # Verify edges from complete portion of the graph were added
        complete_nodes = subg.nodes() & set(range(25))
        assert len(complete_nodes) > 0, f"no complete nodes found in subgraph"
        for n in complete_nodes:
            assert (
                len(subg[n]) == len(complete_nodes) - 1
            )  # definition of complete graph

    results = MultiVerify(dpr).compute("subgraph.sample.node_sampling", graph, 0.4)
    results.normalize(dpr.wrappers.Graph.NetworkXGraph).custom_compare(cmp_func)


def test_edge_sampling(default_plugin_resolver):
    dpr = default_plugin_resolver
    # Build a complete graph, then add a bunch of disconnected nodes
    # Edge Sampling should not pick any of the disconnected nodes for the subgraph
    # For the nodes attached to chosen edges, additional edges should not be added to the subgraph
    g = nx.complete_graph(25)
    g.add_nodes_from(range(25, 50))
    graph = dpr.wrappers.Graph.NetworkXGraph(g)

    def cmp_func(subgraph):
        subg = subgraph.value
        assert 0 < len(subg.nodes()) < len(g.nodes()), f"# nodes = {len(subg.nodes())}"
        # Verify none of the isolated nodes were chosen
        assert not subg.nodes() & set(
            range(25, 50)
        ), f"isolated nodes found in subgraph"
        # Verify not all edges from complete portion of the graph were added
        possible_edges = len(subg.nodes()) - 1
        for n in subg.nodes():
            assert len(subg[n]) < possible_edges, f"all possible edges were added"

    results = MultiVerify(dpr).compute("subgraph.sample.edge_sampling", graph, 0.4)
    results.normalize(dpr.wrappers.Graph.NetworkXGraph).custom_compare(cmp_func)


def test_totally_induced_edge_sampling(default_plugin_resolver):
    dpr = default_plugin_resolver
    # Build a complete graph, then add a bunch of disconnected nodes
    # TIES should not pick any of the disconnected nodes for the subgraph
    # For the nodes attached to chosen edges, all additional edges should be added to the subgraph
    g = nx.complete_graph(25)
    g.add_nodes_from(range(25, 50))
    graph = dpr.wrappers.Graph.NetworkXGraph(g)

    def cmp_func(subgraph):
        subg = subgraph.value
        assert 0 < len(subg.nodes()) < len(g.nodes()), f"# nodes = {len(subg.nodes())}"
        # Verify none of the isolated nodes were chosen
        assert not subg.nodes() & set(
            range(25, 50)
        ), f"isolated nodes found in subgraph"
        # Verify all edges from complete portion of the graph were added
        possible_edges = len(subg.nodes()) - 1
        for n in subg.nodes():
            assert len(subg[n]) == possible_edges, f"not all possible edges were added"

    results = MultiVerify(dpr).compute("subgraph.sample.ties", graph, 0.4)
    results.normalize(dpr.wrappers.Graph.NetworkXGraph).custom_compare(cmp_func)


def test_random_walk_sampling_1(default_plugin_resolver):
    dpr = default_plugin_resolver
    # Build a long chain so random sampling has no randomness
    g = nx.Graph()
    for i in range(50):
        g.add_edge(i, i + 1)
    graph = dpr.wrappers.Graph.NetworkXGraph(g)

    def cmp_func(subgraph):
        subg = subgraph.value
        assert set(subg.nodes()) == set(range(21))

    results = MultiVerify(dpr).compute(
        "subgraph.sample.random_walk",
        graph,
        num_edges=20,
        start_node=0,
        jump_probability=0.015,
    )
    results.normalize(dpr.wrappers.Graph.NetworkXGraph).custom_compare(cmp_func)


def test_random_walk_sampling_2(default_plugin_resolver):
    dpr = default_plugin_resolver
    # Build two disconnected components. Randomly sampling should never leave the starting component.
    # Keep going until all nodes in the starting component have been visited
    g1 = nx.complete_graph(7)
    g2 = nx.complete_graph(range(10, 17))
    g = nx.Graph()
    g.update(g1)
    g.update(g2)
    graph = dpr.wrappers.Graph.NetworkXGraph(g)

    def cmp_func(subgraph):
        subg = subgraph.value
        assert set(subg.nodes()) == set(range(10, 17))

    results = MultiVerify(dpr).compute(
        "subgraph.sample.random_walk", graph, num_nodes=7, start_node=12
    )
    results.normalize(dpr.wrappers.Graph.NetworkXGraph).custom_compare(cmp_func)
