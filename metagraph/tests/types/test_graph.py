import pytest

from metagraph.plugins.networkx.types import NetworkXGraph
from metagraph.plugins.scipy.types import ScipyGraph, ScipyEdgeMap
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
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 333), (2, 1, 3)]
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
        ScipyGraph(g_int, [0, 2, 7]),
        ScipyGraph(g_int, [0, 2, 7]),
        aprops,
        aprops,
        {},
        {},
    )
    with pytest.raises(AssertionError):
        ScipyGraph.Type.assert_equal(
            ScipyGraph(g_int, [0, 2, 7]),
            ScipyGraph(g_int, [0, 1, 2]),
            aprops,
            aprops,
            {},
            {},
        )
    # Node weights affect comparison
    nodes1 = np.array([10, 20, 30])
    nodes2 = np.array([10, 20, 33])
    ScipyGraph.Type.assert_equal(
        ScipyGraph(g_int, node_vals=nodes1),
        ScipyGraph(g_int, node_vals=nodes1),
        {**aprops, "node_type": "map"},
        {**aprops, "node_type": "map"},
        {},
        {},
    )
    ScipyGraph.Type.assert_equal(
        ScipyGraph(g_int, [0, 2, 7], node_vals=nodes1),
        ScipyGraph(g_int, [0, 2, 7], node_vals=nodes1),
        {**aprops, "node_type": "map"},
        {**aprops, "node_type": "map"},
        {},
        {},
    )
    with pytest.raises(AssertionError):
        ScipyGraph.Type.assert_equal(
            ScipyGraph(g_int, node_vals=nodes1),
            ScipyGraph(g_int, node_vals=nodes2),
            {**aprops, "node_type": "map"},
            {**aprops, "node_type": "map"},
            {},
            {},
        )


def test_graphblas():
    grblas = pytest.importorskip("grblas")
    from metagraph.plugins.graphblas.types import GrblasGraph

    # [1 2  ]
    # [  0 3]
    # [  3  ]
    aprops = {
        "is_directed": True,
        "node_type": "set",
        "edge_type": "map",
        "edge_dtype": "int",
    }
    m_int = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3], nrows=3, ncols=3, dtype=int
    )
    m_float = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3], nrows=3, ncols=3, dtype=float
    )
    GrblasGraph.Type.assert_equal(
        GrblasGraph(m_int), GrblasGraph(m_int.dup()), aprops, aprops, {}, {}
    )
    m_close = m_float.dup()
    m_close[0, 0] << 1.0000000000001
    GrblasGraph.Type.assert_equal(
        GrblasGraph(m_close),
        GrblasGraph(m_float),
        {**aprops, "edge_dtype": "float"},
        {**aprops, "edge_dtype": "float"},
        {},
        {},
    )
    m_diff = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 3, 0, 3, 3], nrows=3, ncols=3, dtype=int
    )
    #                                                                       ^^^ changed
    with pytest.raises(AssertionError):
        GrblasGraph.Type.assert_equal(
            GrblasGraph(m_int), GrblasGraph(m_diff), aprops, aprops, {}, {}
        )
    with pytest.raises(AssertionError):
        GrblasGraph.Type.assert_equal(
            GrblasGraph(m_int),
            GrblasGraph(
                grblas.Matrix.from_values(
                    [0, 0, 1, 1, 2], [0, 1, 1, 2, 0], [1, 2, 0, 3, 3], dtype=int
                )
                # change is here                                       ^^^
            ),
            aprops,
            aprops,
            {},
            {},
        )
    with pytest.raises(AssertionError):
        GrblasGraph.Type.assert_equal(
            GrblasGraph(m_int),
            GrblasGraph(
                grblas.Matrix.from_values(
                    [0, 0, 1, 1, 2, 2],
                    [0, 1, 1, 2, 1, 2],
                    [1, 2, 0, 3, 3, 0],
                    dtype=int,
                )
                # extra element                          ^^^                 ^^^                 ^^^
            ),
            aprops,
            aprops,
            {},
            {},
        )
    # Active nodes affects comparison
    m_sparse = grblas.Matrix.new(dtype=m_int.dtype, nrows=8, ncols=8)
    m_sparse[[0, 2, 7], [0, 2, 7]] << m_int
    active = grblas.Vector.from_values([0, 2, 7], [True, True, True], size=8)
    GrblasGraph.Type.assert_equal(
        GrblasGraph(m_sparse, active),
        GrblasGraph(m_sparse, active),
        aprops,
        aprops,
        {},
        {},
    )
    active_extra = grblas.Vector.from_values(
        [0, 2, 4, 7], [True, True, True, True], size=8
    )
    with pytest.raises(AssertionError):
        GrblasGraph.Type.assert_equal(
            GrblasGraph(m_sparse, active),
            GrblasGraph(m_sparse, active_extra),
            aprops,
            aprops,
            {},
            {},
        )
    # The active mask only looks at structure, not values. So this next test should fail.
    active_mixed_boolean = grblas.Vector.from_values(
        [0, 2, 4, 7], [True, True, False, True], size=8
    )
    with pytest.raises(AssertionError):
        GrblasGraph.Type.assert_equal(
            GrblasGraph(m_sparse, active),
            GrblasGraph(m_sparse, active_mixed_boolean),
            aprops,
            aprops,
            {},
            {},
        )
    m_sparse_other = grblas.Matrix.new(dtype=m_int.dtype, nrows=8, ncols=8)
    m_sparse_other[[0, 5, 7], [0, 5, 7]] << m_int
    active_other = grblas.Vector.from_values([0, 5, 7], [True, True, True], size=8)
    with pytest.raises(AssertionError):
        GrblasGraph.Type.assert_equal(
            GrblasGraph(m_sparse, active),
            GrblasGraph(m_sparse_other, active_other),
            aprops,
            aprops,
            {},
            {},
        )
    # Test different size matrices
    m_sparse_big = grblas.Matrix.new(dtype=m_int.dtype, nrows=28, ncols=28)
    m_sparse_big[[0, 2, 7], [0, 2, 7]] << m_int
    active_big = grblas.Vector.from_values([0, 2, 7], [True, True, True], size=28)
    GrblasGraph.Type.assert_equal(
        GrblasGraph(m_sparse_big, nodes=active_big),
        GrblasGraph(m_sparse, nodes=active),
        aprops,
        aprops,
        {},
        {},
    )
    # Node weights affect comparison
    nodes1 = grblas.Vector.from_values([0, 1, 2], [10, 20, 30])
    nodes2 = grblas.Vector.from_values([0, 1, 2], [10, 20, 33])
    GrblasGraph.Type.assert_equal(
        GrblasGraph(m_int, nodes1),
        GrblasGraph(m_int, nodes1),
        {**aprops, "node_type": "map"},
        {**aprops, "node_type": "map"},
        {},
        {},
    )
    with pytest.raises(AssertionError):
        GrblasGraph.Type.assert_equal(
            GrblasGraph(m_int, nodes=nodes1),
            GrblasGraph(m_int, nodes=nodes2),
            {**aprops, "node_type": "map"},
            {**aprops, "node_type": "map"},
            {},
            {},
        )
