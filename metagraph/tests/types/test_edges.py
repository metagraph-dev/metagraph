import pytest
from metagraph.plugins.networkx.types import NetworkXEdgeMap, NetworkXEdgeSet
from metagraph.plugins.pandas.types import PandasEdgeMap, PandasEdgeSet
from metagraph.plugins.graphblas.types import GrblasEdgeMap
from metagraph.plugins.scipy.types import ScipyEdgeMap
from metagraph import NodeLabels
import networkx as nx
import pandas as pd
import grblas
import scipy.sparse as ss
import numpy as np


def test_networkx():
    # 0 -> 0 (weight=1)
    # 0 -> 1 (weight=2)
    # 1 -> 1 (weight=0)
    # 1 -> 2 (weight=3)
    # 2 -> 1 (weight=3)
    g_int = nx.DiGraph()
    g_int.add_weighted_edges_from(
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3), (2, 1, 3)]
    )
    g_float = nx.DiGraph()
    g_float.add_weighted_edges_from(
        [(0, 0, 1.0), (0, 1, 2.0), (1, 1, 0.0), (1, 2, 3.0), (2, 1, 3.0),]
    )
    NetworkXEdgeMap.Type.assert_equal(
        NetworkXEdgeMap(g_int), NetworkXEdgeMap(g_int.copy()), {}, {}, {}, {}
    )
    g_close = g_float.copy()
    g_close.edges[(0, 0)]["weight"] = 1.0000000000001
    NetworkXEdgeMap.Type.assert_equal(
        NetworkXEdgeMap(g_close), NetworkXEdgeMap(g_float), {}, {}, {}, {}
    )
    g_diff1 = nx.DiGraph()
    g_diff1.add_weighted_edges_from(
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3), (2, 1, 333)]
    )
    with pytest.raises(AssertionError):
        NetworkXEdgeMap.Type.assert_equal(
            NetworkXEdgeMap(g_int), NetworkXEdgeMap(g_diff1), {}, {}, {}, {}
        )
    # Edgesets ignore weights
    NetworkXEdgeSet.Type.assert_equal(
        NetworkXEdgeSet(g_int), NetworkXEdgeSet(g_diff1), {}, {}, {}, {}
    )
    g_diff2 = nx.DiGraph()
    g_diff2.add_weighted_edges_from(
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3), (2, 0, 3)]
    )
    with pytest.raises(AssertionError):
        NetworkXEdgeMap.Type.assert_equal(
            NetworkXEdgeMap(g_int), NetworkXEdgeMap(g_diff2), {}, {}, {}, {}
        )
    g_extra = nx.DiGraph()
    g_extra.add_weighted_edges_from(
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3), (2, 1, 3), (2, 0, 2),]
    )
    with pytest.raises(AssertionError):
        NetworkXEdgeMap.Type.assert_equal(
            NetworkXEdgeMap(g_int), NetworkXEdgeMap(g_extra), {}, {}, {}, {}
        )
    # Undirected vs Directed
    g_undir = nx.Graph()
    g_undir.add_weighted_edges_from([(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3)])
    g_dir = nx.DiGraph()
    g_dir.add_weighted_edges_from([(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3)])
    with pytest.raises(AssertionError):
        NetworkXEdgeMap.Type.assert_equal(
            NetworkXEdgeMap(g_undir),
            NetworkXEdgeMap(g_dir),
            {"is_directed": False},
            {"is_directed": True},
            {},
            {},
        )
    NetworkXEdgeMap.Type.assert_equal(
        NetworkXEdgeMap(g_undir), NetworkXEdgeMap(g_undir), {}, {}, {}, {},
    )
    # Different weight_label
    g_wgt = nx.DiGraph()
    g_wgt.add_weighted_edges_from(
        [(0, 0, 1), (0, 1, 2), (1, 1, 0), (1, 2, 3), (2, 1, 3)], weight="WGT",
    )
    NetworkXEdgeMap.Type.assert_equal(
        NetworkXEdgeMap(g_int, weight_label="weight"),
        NetworkXEdgeMap(g_wgt, weight_label="WGT"),
        {},
        {},
        {},
        {},
    )


def test_pandas_edge():
    # A -> A (weight=1)
    # A -> B (weight=2)
    # B -> B (weight=0)
    # B -> C (weight=3)
    # C -> B (weight=3)
    df = pd.DataFrame(
        {
            "source": ["A", "A", "B", "B", "C"],
            "target": ["A", "B", "B", "C", "B"],
            "weight": [1, 2, 0, 3, 3],
        }
    )
    PandasEdgeMap.Type.assert_equal(
        PandasEdgeMap(df), PandasEdgeMap(df.copy()), {}, {}, {}, {}
    )
    df_float = df.copy()
    df_float["weight"] = df_float["weight"].astype(np.float64)
    df_close = df_float.copy()
    df_close.loc[0, "weight"] = 1.0000000000001
    PandasEdgeMap.Type.assert_equal(
        PandasEdgeMap(df_close), PandasEdgeMap(df_float), {}, {}, {}, {}
    )
    diff1 = df.copy()
    diff1.loc[4, "weight"] = 333
    with pytest.raises(AssertionError):
        PandasEdgeMap.Type.assert_equal(
            PandasEdgeMap(df), PandasEdgeMap(diff1), {}, {}, {}, {}
        )
    # Edgesets ignore weights
    PandasEdgeSet.Type.assert_equal(
        PandasEdgeSet(df), PandasEdgeSet(diff1), {}, {}, {}, {}
    )
    diff2 = df.copy()
    diff2.loc[4, "target"] = "A"
    with pytest.raises(AssertionError):
        PandasEdgeMap.Type.assert_equal(
            PandasEdgeMap(df), PandasEdgeMap(diff2), {}, {}, {}, {}
        )
    extra = df.copy()
    extra = extra.append(pd.Series([2], index=["weight"], name=("C", "A")))
    with pytest.raises(AssertionError):
        PandasEdgeMap.Type.assert_equal(
            PandasEdgeMap(df), PandasEdgeMap(extra), {}, {}, {}, {}
        )
    # Undirected vs Directed
    with pytest.raises(AssertionError):
        PandasEdgeMap.Type.assert_equal(
            PandasEdgeMap(df),
            PandasEdgeMap(df, is_directed=False),
            {"is_directed": True},
            {"is_directed": False},
            {},
            {},
        )
    PandasEdgeMap.Type.assert_equal(
        PandasEdgeMap(df, is_directed=False),
        PandasEdgeMap(df, is_directed=False),
        {"is_directed": False},
        {"is_directed": False},
        {},
        {},
    )
    # Different weight_label
    wgt = df.copy()
    wgt = wgt.rename(columns={"weight": "WGT"})
    PandasEdgeMap.Type.assert_equal(
        PandasEdgeMap(df, weight_label="weight"),
        PandasEdgeMap(wgt, weight_label="WGT"),
        {},
        {},
        {},
        {},
    )


def test_graphblas_adj():
    # [1 2  ]
    # [  0 3]
    # [  3  ]
    g_int = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3], dtype=grblas.dtypes.INT64
    )
    g_float = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3], dtype=grblas.dtypes.FP64
    )
    GrblasEdgeMap.Type.assert_equal(
        GrblasEdgeMap(g_int), GrblasEdgeMap(g_int.dup()), {}, {}, {}, {}
    )
    g_close = g_float.dup()
    g_close[0, 0] = 1.0000000000001
    GrblasEdgeMap.Type.assert_equal(
        GrblasEdgeMap(g_close), GrblasEdgeMap(g_float), {}, {}, {}, {}
    )
    g_diff = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 3, 0, 3, 3]
    )  # change is here                     ^^^
    with pytest.raises(AssertionError):
        GrblasEdgeMap.Type.assert_equal(
            GrblasEdgeMap(g_int), GrblasEdgeMap(g_diff), {}, {}, {}, {}
        )
    with pytest.raises(AssertionError):
        GrblasEdgeMap.Type.assert_equal(
            GrblasEdgeMap(g_int),
            GrblasEdgeMap(
                grblas.Matrix.from_values(
                    [0, 0, 1, 1, 2], [0, 1, 1, 2, 0], [1, 2, 0, 3, 3]
                )  # change is here              ^^^
            ),
            {},
            {},
            {},
            {},
        )
    with pytest.raises(AssertionError):
        GrblasEdgeMap.Type.assert_equal(
            GrblasEdgeMap(g_int),
            GrblasEdgeMap(
                grblas.Matrix.from_values(
                    [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 1, 2], [1, 2, 0, 3, 3, 0]
                )  # extra element ^^^                 ^^^                 ^^^
            ),
            {},
            {},
            {},
            {},
        )
    # Transposed
    GrblasEdgeMap.Type.assert_equal(
        GrblasEdgeMap(g_int),
        GrblasEdgeMap(
            grblas.Matrix.from_values(
                [0, 1, 1, 1, 2], [0, 0, 1, 2, 1], [1, 2, 0, 3, 3]
            ),
            transposed=True,
        ),
        {},
        {},
        {},
        {},
    )
    GrblasEdgeMap.Type.assert_equal(
        GrblasEdgeMap(g_int, transposed=True),
        GrblasEdgeMap(
            grblas.Matrix.from_values([0, 1, 1, 1, 2], [0, 0, 1, 2, 1], [1, 2, 0, 3, 3])
        ),
        {},
        {},
        {},
        {},
    )
    GrblasEdgeMap.Type.assert_equal(
        GrblasEdgeMap(g_int, transposed=True),
        GrblasEdgeMap(g_int, transposed=True),
        {},
        {},
        {},
        {},
    )


def test_scipy_adj():
    # [1 2  ]
    # [  0 3]
    # [  3  ]
    g_int = ss.coo_matrix(
        ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1])), dtype=np.int64
    )
    g_float = ss.coo_matrix(
        ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1])), dtype=np.float64
    )
    ScipyEdgeMap.Type.assert_equal(
        ScipyEdgeMap(g_int), ScipyEdgeMap(g_int.copy().tocsr()), {}, {}, {}, {}
    )
    g_close = g_float.tocsr()
    g_close[0, 0] = 1.0000000000001
    ScipyEdgeMap.Type.assert_equal(
        ScipyEdgeMap(g_close), ScipyEdgeMap(g_float), {}, {}, {}, {}
    )
    g_diff = ss.coo_matrix(
        ([1, 3, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1]))
    )  # -  ^^^ changed
    with pytest.raises(AssertionError):
        ScipyEdgeMap.Type.assert_equal(
            ScipyEdgeMap(g_int), ScipyEdgeMap(g_diff), {}, {}, {}, {}
        )
    with pytest.raises(AssertionError):
        ScipyEdgeMap.Type.assert_equal(
            ScipyEdgeMap(g_int),
            ScipyEdgeMap(
                ss.coo_matrix(
                    ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 0]))
                )  # change is here                                 ^^^
            ),
            {},
            {},
            {},
            {},
        )
    with pytest.raises(AssertionError):
        ScipyEdgeMap.Type.assert_equal(
            ScipyEdgeMap(g_int),
            ScipyEdgeMap(
                ss.coo_matrix(
                    ([1, 2, 0, 3, 3, 0], ([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 1, 2]))
                )  # extra element  ^^^                  ^^^                 ^^^
            ),
            {},
            {},
            {},
            {},
        )
    # Node index affects comparison
    ScipyEdgeMap.Type.assert_equal(
        ScipyEdgeMap(g_int, [0, 2, 7]), ScipyEdgeMap(g_int, [0, 2, 7]), {}, {}, {}, {}
    )
    with pytest.raises(AssertionError):
        ScipyEdgeMap.Type.assert_equal(
            ScipyEdgeMap(g_int, [0, 2, 7]),
            ScipyEdgeMap(g_int, [0, 1, 2]),
            {},
            {},
            {},
            {},
        )
    # Transposed
    ScipyEdgeMap.Type.assert_equal(
        ScipyEdgeMap(g_int),
        ScipyEdgeMap(
            ss.coo_matrix(([1, 2, 0, 3, 3], ([0, 1, 1, 1, 2], [0, 0, 1, 2, 1]))),
            transposed=True,
        ),
        {},
        {},
        {},
        {},
    )
    ScipyEdgeMap.Type.assert_equal(
        ScipyEdgeMap(g_int, transposed=True),
        ScipyEdgeMap(
            ss.coo_matrix(([1, 2, 0, 3, 3], ([0, 1, 1, 1, 2], [0, 0, 1, 2, 1])))
        ),
        {},
        {},
        {},
        {},
    )
    ScipyEdgeMap.Type.assert_equal(
        ScipyEdgeMap(g_int, transposed=True),
        ScipyEdgeMap(g_int, transposed=True),
        {},
        {},
        {},
        {},
    )
