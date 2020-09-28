import pytest

grblas = pytest.importorskip("grblas")

from metagraph.plugins.networkx.types import NetworkXGraph
from metagraph.plugins.scipy.types import ScipyGraph, ScipyEdgeMap
from metagraph.plugins.numpy.types import NumpyNodeMap
import networkx as nx
import numpy as np
import scipy.sparse as ss


def test_networkx():
    # 0 -> 0 (weight=1)
    # 0 -> 1 (weight=2)
    # 1 -> 1 (weight=0)
    # 1 -> 2 (weight=3)
    # 2 -> 1 (weight=3)
    aprops = {
        "is_directed": True,
        "node_type": "set",
        "edge_type": "map",
        "edge_dtype": "int",
    }
    g_int = nx.DiGraph()
    g_int.add_weighted_edges_from(
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3), (2, 1, 3)]
    )
    g_float = nx.DiGraph()
    g_float.add_weighted_edges_from(
        [(0, 0, 1.0), (0, 1, 2.0), (1, 1, 0.0), (1, 2, 3.0), (2, 1, 3.0),]
    )
    NetworkXGraph.Type.assert_equal(
        NetworkXGraph(g_int), NetworkXGraph(g_int.copy()), aprops, aprops, {}, {}
    )
    g_close = g_float.copy()
    g_close.edges[(0, 0)]["weight"] = 1.0000000000001
    NetworkXGraph.Type.assert_equal(
        NetworkXGraph(g_close),
        NetworkXGraph(g_float),
        {**aprops, "edge_dtype": "float"},
        {**aprops, "edge_dtype": "float"},
        {},
        {},
    )
    g_diff1 = nx.DiGraph()
    g_diff1.add_weighted_edges_from(
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3), (2, 1, 333)]
    )
    with pytest.raises(AssertionError):
        NetworkXGraph.Type.assert_equal(
            NetworkXGraph(g_int), NetworkXGraph(g_diff1), aprops, aprops, {}, {}
        )
    g_diff2 = nx.DiGraph()
    g_diff2.add_weighted_edges_from(
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3), (2, 0, 3)]
    )
    with pytest.raises(AssertionError):
        NetworkXGraph.Type.assert_equal(
            NetworkXGraph(g_int), NetworkXGraph(g_diff2), aprops, aprops, {}, {}
        )
    g_extra = nx.DiGraph()
    g_extra.add_weighted_edges_from(
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3), (2, 1, 3), (2, 0, 2),]
    )
    with pytest.raises(AssertionError):
        NetworkXGraph.Type.assert_equal(
            NetworkXGraph(g_int), NetworkXGraph(g_extra), aprops, aprops, {}, {}
        )
    # Undirected vs Directed
    g_undir = nx.Graph()
    g_undir.add_weighted_edges_from([(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3)])
    g_dir = nx.DiGraph()
    g_dir.add_weighted_edges_from([(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3)])
    with pytest.raises(AssertionError):
        NetworkXGraph.Type.assert_equal(
            NetworkXGraph(g_undir),
            NetworkXGraph(g_dir),
            {**aprops, "is_directed": False},
            aprops,
            {},
            {},
        )
    NetworkXGraph.Type.assert_equal(
        NetworkXGraph(g_undir),
        NetworkXGraph(g_undir),
        {**aprops, "is_directed": False},
        {**aprops, "is_directed": False},
        {},
        {},
    )
    # Different weight_label
    g_wgt = nx.DiGraph()
    g_wgt.add_weighted_edges_from(
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3), (2, 1, 3)], weight="WGT",
    )
    NetworkXGraph.Type.assert_equal(
        NetworkXGraph(g_int, edge_weight_label="weight"),
        NetworkXGraph(g_wgt, edge_weight_label="WGT"),
        aprops,
        aprops,
        {},
        {},
    )


def test_scipy():
    # [1 2  ]
    # [  0 3]
    # [  3  ]
    aprops = {
        "is_directed": True,
        "node_type": "set",
        "edge_type": "map",
        "edge_dtype": "int",
    }
    g_int = ss.coo_matrix(
        ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1])), dtype=np.int64
    )
    g_float = ss.coo_matrix(
        ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1])), dtype=np.float64
    )
    ScipyGraph.Type.assert_equal(
        ScipyGraph(g_int), ScipyGraph(g_int.copy().tocsr()), aprops, aprops, {}, {}
    )
    g_close = g_float.tocsr()
    g_close[0, 0] = 1.0000000000001
    ScipyGraph.Type.assert_equal(
        ScipyGraph(g_close),
        ScipyGraph(g_float),
        {**aprops, "edge_dtype": "float"},
        {**aprops, "edge_dtype": "float"},
        {},
        {},
    )
    g_diff = ss.coo_matrix(
        ([1, 3, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1]))
    )  # -  ^^^ changed
    with pytest.raises(AssertionError):
        ScipyGraph.Type.assert_equal(
            ScipyGraph(g_int), ScipyGraph(g_diff), aprops, aprops, {}, {}
        )
    with pytest.raises(AssertionError):
        ScipyGraph.Type.assert_equal(
            ScipyGraph(g_int),
            ScipyGraph(
                ss.coo_matrix(
                    ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 0]))
                )  # change is here                                 ^^^
            ),
            aprops,
            aprops,
            {},
            {},
        )
    with pytest.raises(AssertionError):
        ScipyGraph.Type.assert_equal(
            ScipyGraph(g_int),
            ScipyGraph(
                ss.coo_matrix(
                    ([1, 2, 0, 3, 3, 0], ([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 1, 2]))
                )  # extra element  ^^^                  ^^^                 ^^^
            ),
            aprops,
            aprops,
            {},
            {},
        )
    # Node index affects comparison
    ScipyGraph.Type.assert_equal(
        ScipyGraph(ScipyEdgeMap(g_int, [0, 2, 7])),
        ScipyGraph(ScipyEdgeMap(g_int, [0, 2, 7])),
        aprops,
        aprops,
        {},
        {},
    )
    with pytest.raises(AssertionError):
        ScipyGraph.Type.assert_equal(
            ScipyGraph(ScipyEdgeMap(g_int, [0, 2, 7])),
            ScipyGraph(ScipyEdgeMap(g_int, [0, 1, 2])),
            aprops,
            aprops,
            {},
            {},
        )
    # Node weights affect comparison
    nodes1 = NumpyNodeMap(np.array([10, 20, 30]))
    nodes2 = NumpyNodeMap(np.array([10, 20, 33]))
    ScipyGraph.Type.assert_equal(
        ScipyGraph(ScipyEdgeMap(g_int), nodes1),
        ScipyGraph(ScipyEdgeMap(g_int), nodes1),
        {**aprops, "node_type": "map"},
        {**aprops, "node_type": "map"},
        {},
        {},
    )
    with pytest.raises(AssertionError):
        ScipyGraph.Type.assert_equal(
            ScipyGraph(ScipyEdgeMap(g_int), nodes1),
            ScipyGraph(ScipyEdgeMap(g_int), nodes2),
            {**aprops, "node_type": "map"},
            {**aprops, "node_type": "map"},
            {},
            {},
        )
