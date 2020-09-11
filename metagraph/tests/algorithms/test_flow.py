from metagraph.tests.util import default_plugin_resolver
import metagraph as mg
import networkx as nx
from . import MultiVerify
from typing import Tuple
import math


def test_max_flow(default_plugin_resolver):
    """
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

    ebunch_answer = [
        (0, 1, 0),
        (0, 3, 6),
        (1, 4, 0),
        (2, 7, 6),
        (3, 1, 0),
        (3, 4, 6),
        (4, 2, 0),
        (4, 5, 6),
        (5, 2, 5),
        (5, 6, 1),
        (6, 2, 1),
    ]
    nx_graph_answer = nx.DiGraph()
    nx_graph_answer.add_weighted_edges_from(ebunch_answer)
    expected_flow_value = 6
    expected_graph = mg.plugins.networkx.types.NetworkXGraph(nx_graph_answer)

    def cmp_func(x):
        actual_flow_value, actual_flow_graph = x

        rel_tol = 1e-9
        abs_tol = 0.0
        assert math.isclose(
            actual_flow_value, expected_flow_value, rel_tol=rel_tol, abs_tol=abs_tol
        )

        assert actual_flow_graph.value.nodes() == expected_graph.value.nodes()

        bottleneck_nodes = [4, 2]
        get_edge_weight = lambda edge: edge[2]
        for node in bottleneck_nodes:
            actual_total_in_flow = sum(
                map(
                    get_edge_weight,
                    actual_flow_graph.value.in_edges(
                        node, data=expected_graph.edge_weight_label
                    ),
                )
            )
            expected_total_in_flow = sum(
                map(
                    get_edge_weight,
                    expected_graph.value.in_edges(
                        node, data=expected_graph.edge_weight_label
                    ),
                )
            )
            assert actual_total_in_flow == expected_total_in_flow
            actual_total_out_flow = sum(
                map(
                    get_edge_weight,
                    actual_flow_graph.value.out_edges(
                        node, data=expected_graph.edge_weight_label
                    ),
                )
            )
            expected_total_out_flow = sum(
                map(
                    get_edge_weight,
                    expected_graph.value.out_edges(
                        node, data=expected_graph.edge_weight_label
                    ),
                )
            )
            assert actual_total_out_flow == expected_total_out_flow

    MultiVerify(dpr).compute(
        "flow.max_flow", graph, source_node, target_node
    ).normalize((float, dpr.wrappers.Graph.NetworkXGraph.Type)).custom_compare(cmp_func)
