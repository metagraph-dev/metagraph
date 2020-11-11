from metagraph.tests.util import default_plugin_resolver
import metagraph as mg
import networkx as nx
from . import MultiVerify


def test_max_flow(default_plugin_resolver):
    r"""
    0 ---9-> 1        5 --1--> 6
    |      ^ |      ^ |      /
    |     /  |     /  |     /
    10   2   3    7   5   11
    |  _/    |  /     |   /
    v /      v /      v v
    3 --8--> 4 ---4-> 2 --6--> 7
    """
    dpr = default_plugin_resolver
    source_node = 0
    target_node = 7
    ebunch = [
        (0, 1, 9),
        (0, 3, 10),
        (1, 4, 3),
        (2, 7, 6),
        (3, 1, 2),
        (3, 4, 8),
        (4, 5, 7),
        (4, 2, 4),
        (5, 2, 5),
        (5, 6, 1),
        (6, 2, 11),
    ]
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph, edge_weight_label="weight")

    expected_flow_value = 6
    bottleneck_nodes = {2, 4}
    expected_nodemap = {2: 6, 4: 6}

    mv = MultiVerify(dpr)
    results = mv.compute("flow.max_flow", graph, source_node, target_node)

    # Compare flow rate
    results[0].assert_equal(expected_flow_value)

    # Normalize actual flow to prepare to transform
    actual_flow = results[1].normalize(dpr.wrappers.Graph.NetworkXGraph)

    # Compare sum of out edges for bottleneck nodes
    out_edges = mv.transform(
        dpr.plugins.core_networkx.algos.util.graph.aggregate_edges,
        actual_flow,
        lambda x, y: x + y,
        initial_value=0,
    )
    out_bottleneck = mv.transform(
        dpr.algos.util.nodemap.select.core_python, out_edges, bottleneck_nodes
    )
    out_bottleneck.assert_equal(expected_nodemap)

    # Compare sum of in edges for bottleneck nodes
    in_edges = mv.transform(
        "util.graph.aggregate_edges.core_networkx",
        actual_flow,
        lambda x, y: x + y,
        initial_value=0,
        in_edges=True,
        out_edges=False,
    )
    in_bottleneck = mv.transform(
        "util.nodemap.select.core_python", in_edges, bottleneck_nodes
    )
    in_bottleneck.assert_equal(expected_nodemap)


def test_min_cut(default_plugin_resolver):
    r"""
    0 ---9-> 1        5 --1--> 6
    |      ^ |      ^ |      /
    |     /  |     /  |     /
    10   2   3    1   5   11
    |  _/    |  /     |   /
    v /      v /      v v
    3 --8--> 4 ---4-> 2 --6--> 7
    """
    dpr = default_plugin_resolver
    source_node = 0
    target_node = 7
    ebunch = [
        (0, 1, 9),
        (0, 3, 10),
        (1, 4, 3),
        (2, 7, 6),
        (3, 1, 2),
        (3, 4, 8),
        (4, 5, 1),
        (4, 2, 4),
        (5, 2, 5),
        (5, 6, 1),
        (6, 2, 11),
    ]
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph, edge_weight_label="weight")

    expected_flow_value = 5
    cut_edges = nx.DiGraph()
    cut_edges.add_nodes_from(nx_graph.nodes)
    cut_edges.add_weighted_edges_from([(4, 5, 1), (4, 2, 4)])
    expected_cut_edges = dpr.wrappers.Graph.NetworkXGraph(cut_edges)

    mv = MultiVerify(dpr)
    results = mv.compute("flow.min_cut", graph, source_node, target_node)

    # Compare flow rate
    results[0].assert_equal(expected_flow_value)

    # Compare cut graph
    results[1].assert_equal(expected_cut_edges)
