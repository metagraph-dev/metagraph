import pytest

from metagraph.plugins.networkx.types import NetworkXBipartiteGraph
import networkx as nx


def test_networkx():
    # 0 -> 11 (weight=1)
    # 0 -> 12 (weight=2)
    # 1 -> 11 (weight=0)
    # 1 -> 12 (weight=3)
    # 1 -> 13 (weight=3)
    aprops = {
        "is_directed": True,
        "node0_type": "set",
        "node1_type": "set",
        "edge_type": "map",
        "edge_dtype": "int",
    }
    g_int = nx.Graph()
    g_int.add_weighted_edges_from(
        [(0, 11, 1), (0, 12, 2), (1, 11, 0), (1, 12, 3), (1, 13, 3)]
    )
    g_float = nx.Graph()
    g_float.add_weighted_edges_from(
        [(0, 11, 1.0), (0, 12, 2.0), (1, 11, 0.0), (1, 12, 3.0), (1, 13, 3.0)]
    )
    nodes = ({0, 1}, {11, 12, 13})
    NetworkXBipartiteGraph.Type.assert_equal(
        NetworkXBipartiteGraph(g_int, nodes),
        NetworkXBipartiteGraph(g_int.copy(), nodes),
        aprops,
        aprops,
        {},
        {},
    )
    g_close = g_float.copy()
    g_close.edges[(0, 11)]["weight"] = 1.0000000000001
    NetworkXBipartiteGraph.Type.assert_equal(
        NetworkXBipartiteGraph(g_close, nodes),
        NetworkXBipartiteGraph(g_float, nodes),
        {**aprops, "edge_dtype": "float"},
        {**aprops, "edge_dtype": "float"},
        {},
        {},
    )
    g_diff1 = nx.Graph()
    g_diff1.add_weighted_edges_from(
        [(0, 11, 1), (0, 12, 2), (1, 11, 0), (1, 12, 333), (1, 13, 3)]
    )
    with pytest.raises(AssertionError):
        NetworkXBipartiteGraph.Type.assert_equal(
            NetworkXBipartiteGraph(g_int, nodes),
            NetworkXBipartiteGraph(g_diff1, nodes),
            aprops,
            aprops,
            {},
            {},
        )
    g_diff2 = nx.Graph()
    g_diff2.add_weighted_edges_from(
        [(0, 11, 1), (0, 12, 2), (1, 11, 0), (1, 12, 3), (0, 13, 3)]
    )
    with pytest.raises(AssertionError):
        NetworkXBipartiteGraph.Type.assert_equal(
            NetworkXBipartiteGraph(g_int, nodes),
            NetworkXBipartiteGraph(g_diff2, nodes),
            aprops,
            aprops,
            {},
            {},
        )
    g_extra = nx.Graph()
    g_extra.add_weighted_edges_from(
        [(0, 11, 1), (0, 12, 2), (1, 11, 0), (1, 12, 3), (1, 13, 3), (0, 13, 2)]
    )
    with pytest.raises(AssertionError):
        NetworkXBipartiteGraph.Type.assert_equal(
            NetworkXBipartiteGraph(g_int, nodes),
            NetworkXBipartiteGraph(g_extra, nodes),
            aprops,
            aprops,
            {},
            {},
        )
    # Different nodes
    with pytest.raises(AssertionError):
        NetworkXBipartiteGraph.Type.assert_equal(
            NetworkXBipartiteGraph(g_int, nodes),
            NetworkXBipartiteGraph(g_int, ({0, 1, 2, 3, 4, 5, 6}, {11, 12, 13})),
            aprops,
            aprops,
            {},
            {},
        )
    # Different weight_label
    g_wgt = nx.Graph()
    g_wgt.add_weighted_edges_from(
        [(0, 11, 1), (0, 12, 2), (1, 11, 0), (1, 12, 3), (1, 13, 3)], weight="WGT"
    )
    NetworkXBipartiteGraph.Type.assert_equal(
        NetworkXBipartiteGraph(g_int, nodes, edge_weight_label="weight"),
        NetworkXBipartiteGraph(g_wgt, nodes, edge_weight_label="WGT"),
        aprops,
        aprops,
        {},
        {},
    )
    # Node weights
    g_nodes = g_int.copy()
    nx.set_node_attributes(g_nodes, 1, "nwgt_int")
    nx.set_node_attributes(g_nodes, 1.1, "nwgt_float")
    graph_int = NetworkXBipartiteGraph(g_nodes, nodes, node_weight_label="nwgt_int")
    graph_float = NetworkXBipartiteGraph(g_nodes, nodes, node_weight_label="nwgt_float")
    NetworkXBipartiteGraph.Type.get_type(graph_int)
    NetworkXBipartiteGraph.Type.get_type(graph_float)
    aprops_int = NetworkXBipartiteGraph.Type.get_typeinfo(
        graph_int
    ).known_abstract_props
    aprops_float = NetworkXBipartiteGraph.Type.get_typeinfo(
        graph_float
    ).known_abstract_props
    NetworkXBipartiteGraph.Type.assert_equal(
        graph_int, graph_int, aprops_int, aprops_int, {}, {}
    )
    NetworkXBipartiteGraph.Type.assert_equal(
        graph_float, graph_float, aprops_float, aprops_float, {}, {}
    )

    # Exercise NetworkXBipartiteGraph
    with pytest.raises(ValueError, match="Node IDs found in both parts of the graph"):
        NetworkXBipartiteGraph(g_int, ({0, 1, 11}, {11, 12, 13}))
    with pytest.raises(TypeError, match="Directed Graph not supported"):
        g = nx.DiGraph()
        NetworkXBipartiteGraph(g, nodes)
    with pytest.raises(TypeError, match="nodes must have length of 2"):
        NetworkXBipartiteGraph(g_int, ({0, 1}, {0, 1, 2, 3}, {0, 1}))
    with pytest.raises(
        ValueError, match="Node IDs found in graph, but not listed in either part"
    ):
        NetworkXBipartiteGraph(g_int, ({0, 1}, {11, 12}))
