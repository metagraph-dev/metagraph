import pytest
from metagraph.plugins.networkx.types import NetworkXGraph
from metagraph.plugins.pandas.types import PandasEdgeList
from metagraph.plugins.graphblas.types import GrblasAdjacencyMatrix
from metagraph.plugins.scipy.types import ScipyAdjacencyMatrix
from metagraph import IndexedNodes
import networkx as nx
import pandas as pd
import grblas
import scipy.sparse as ss
import numpy as np


def test_networkx():
    # A -> A (weight=1)
    # A -> B (weight=2)
    # B -> B (weight=0)
    # B -> C (weight=3)
    # C -> B (weight=3)
    g_int = nx.DiGraph()
    g_int.add_weighted_edges_from(
        [("A", "A", 1), ("A", "B", 2), ("B", "B", 0), ("B", "C", 3), ("C", "B", 3)]
    )
    g_float = nx.DiGraph()
    g_float.add_weighted_edges_from(
        [
            ("A", "A", 1.0),
            ("A", "B", 2.0),
            ("B", "B", 0.0),
            ("B", "C", 3.0),
            ("C", "B", 3.0),
        ]
    )
    assert NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_int, weight_label="weight"),
        NetworkXGraph(g_int.copy(), weight_label="weight"),
    )
    g_close = g_float.copy()
    g_close.edges[("A", "A")]["weight"] = 1.0000000000001
    assert NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_close, weight_label="weight"),
        NetworkXGraph(g_float, weight_label="weight"),
    )
    g_diff1 = nx.DiGraph()
    g_diff1.add_weighted_edges_from(
        [("A", "A", 1), ("A", "B", 2), ("B", "B", 0), ("B", "C", 3), ("C", "B", 333)]
    )
    assert not NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_int, weight_label="weight"),
        NetworkXGraph(g_diff1, weight_label="weight"),
    )
    # Ignore weights if unweighted
    assert NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_int), NetworkXGraph(g_diff1)
    )
    g_diff2 = nx.DiGraph()
    g_diff2.add_weighted_edges_from(
        [("A", "A", 1), ("A", "B", 2), ("B", "B", 0), ("B", "C", 3), ("C", "A", 3)]
    )
    assert not NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_int, weight_label="weight"),
        NetworkXGraph(g_diff2, weight_label="weight"),
    )
    g_extra = nx.DiGraph()
    g_extra.add_weighted_edges_from(
        [
            ("A", "A", 1),
            ("A", "B", 2),
            ("B", "B", 0),
            ("B", "C", 3),
            ("C", "B", 3),
            ("C", "A", 2),
        ]
    )
    assert not NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_int, weight_label="weight"),
        NetworkXGraph(g_extra, weight_label="weight"),
    )
    # weights don't match, so we take the fast path and declare them not equal
    assert not NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_int, weight_label="weight"),
        NetworkXGraph(g_int),  # not providing weight_label will assume unweighted graph
    )
    # Undirected vs Directed
    g_undir = nx.Graph()
    g_undir.add_weighted_edges_from(
        [("A", "A", 1), ("A", "B", 2), ("B", "B", 0), ("B", "C", 3)]
    )
    g_dir = nx.DiGraph()
    g_dir.add_weighted_edges_from(
        [("A", "A", 1), ("A", "B", 2), ("B", "B", 0), ("B", "C", 3)]
    )
    assert not NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_undir, weight_label="weight"),
        NetworkXGraph(g_dir, weight_label="weight"),
    )
    assert NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_undir, weight_label="weight"),
        NetworkXGraph(g_undir, weight_label="weight"),
    )
    # Different weight_label
    g_wgt = nx.DiGraph()
    g_wgt.add_weighted_edges_from(
        [("A", "A", 1), ("A", "B", 2), ("B", "B", 0), ("B", "C", 3), ("C", "B", 3)],
        weight="WGT",
    )
    assert NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_int, weight_label="weight"),
        NetworkXGraph(g_wgt, weight_label="WGT"),
    )
    # Node index has no effect
    assert NetworkXGraph.Type.compare_objects(
        NetworkXGraph(g_int, weight_label="weight", node_index=IndexedNodes("ABC")),
        NetworkXGraph(g_int, weight_label="weight", node_index=IndexedNodes("BCA")),
    )
    with pytest.raises(TypeError):
        NetworkXGraph.Type.compare_objects(5, 5)


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
    assert PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df, weight_label="weight"),
        PandasEdgeList(df.copy(), weight_label="weight"),
    )
    df_float = df.copy()
    df_float["weight"] = df_float["weight"].astype(np.float64)
    df_close = df_float.copy()
    df_close.loc[0, "weight"] = 1.0000000000001
    assert PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df_close, weight_label="weight"),
        PandasEdgeList(df_float, weight_label="weight"),
    )
    diff1 = df.copy()
    diff1.loc[4, "weight"] = 333
    assert not PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df, weight_label="weight"),
        PandasEdgeList(diff1, weight_label="weight"),
    )
    # Ignore weights if unweighted
    assert PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df), PandasEdgeList(diff1)
    )
    diff2 = df.copy()
    diff2.loc[4, "target"] = "A"
    assert not PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df, weight_label="weight"),
        PandasEdgeList(diff2, weight_label="weight"),
    )
    extra = df.copy()
    extra = extra.append(pd.Series([2], index=["weight"], name=("C", "A")))
    assert not PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df, weight_label="weight"),
        PandasEdgeList(extra, weight_label="weight"),
    )
    # weights don't match, so we take the fast path and declare them not equal
    assert not PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df, weight_label="weight"),
        PandasEdgeList(df),  # not providing weight_label will assume unweighted graph
    )
    # Undirected vs Directed
    assert not PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df, weight_label="weight"),
        PandasEdgeList(df, weight_label="weight", is_directed=False),
    )
    assert PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df, weight_label="weight", is_directed=False),
        PandasEdgeList(df, weight_label="weight", is_directed=False),
    )
    # Different weight_label
    wgt = df.copy()
    wgt = wgt.rename(columns={"weight": "WGT"})
    assert PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df, weight_label="weight"),
        PandasEdgeList(wgt, weight_label="WGT"),
    )
    # Node index has no effect
    assert PandasEdgeList.Type.compare_objects(
        PandasEdgeList(df, weight_label="weight", node_index=IndexedNodes("ABC")),
        PandasEdgeList(df, weight_label="weight", node_index=IndexedNodes("BCA")),
    )
    with pytest.raises(TypeError):
        PandasEdgeList.Type.compare_objects(5, 5)


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
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_int), GrblasAdjacencyMatrix(g_int.dup()),
    )
    g_close = g_float.dup()
    g_close[0, 0] = 1.0000000000001
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_close), GrblasAdjacencyMatrix(g_float),
    )
    g_diff = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 3, 0, 3, 3]
    )  # change is here                     ^^^
    assert not GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_int), GrblasAdjacencyMatrix(g_diff),
    )
    # Ignore weights if unweighted
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_int, weights="unweighted"),
        GrblasAdjacencyMatrix(g_diff, weights="unweighted"),
    )
    assert not GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_int),
        GrblasAdjacencyMatrix(
            grblas.Matrix.from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 0], [1, 2, 0, 3, 3]
            )  # change is here              ^^^
        ),
    )
    assert not GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_int),
        GrblasAdjacencyMatrix(
            grblas.Matrix.from_values(
                [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 1, 2], [1, 2, 0, 3, 3, 0]
            )  # extra element ^^^                 ^^^                 ^^^
        ),
    )
    # weights don't match, so we take the fast path and declare them not equal
    assert not GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_int), GrblasAdjacencyMatrix(g_int, weights="any"),
    )
    # Node index affects comparison
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_int, node_index=IndexedNodes("ABC")),
        GrblasAdjacencyMatrix(
            grblas.Matrix.from_values(
                [0, 0, 1, 2, 2], [0, 1, 0, 0, 2], [0, 3, 3, 2, 1]
            ),
            node_index=IndexedNodes("BCA"),
        ),
    )
    # Transposed
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_int),
        GrblasAdjacencyMatrix(
            grblas.Matrix.from_values(
                [0, 1, 1, 1, 2], [0, 0, 1, 2, 1], [1, 2, 0, 3, 3]
            ),
            transposed=True,
        ),
    )
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_int, transposed=True),
        GrblasAdjacencyMatrix(
            grblas.Matrix.from_values([0, 1, 1, 1, 2], [0, 0, 1, 2, 1], [1, 2, 0, 3, 3])
        ),
    )
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(g_int, transposed=True),
        GrblasAdjacencyMatrix(g_int, transposed=True),
    )
    with pytest.raises(TypeError):
        GrblasAdjacencyMatrix.Type.compare_objects(5, 5)


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
    assert ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_int), ScipyAdjacencyMatrix(g_int.copy().tocsr()),
    )
    g_close = g_float.tocsr()
    g_close[0, 0] = 1.0000000000001
    assert ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_close), ScipyAdjacencyMatrix(g_float),
    )
    g_diff = ss.coo_matrix(
        ([1, 3, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1]))
    )  # -  ^^^ changed
    assert not ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_int), ScipyAdjacencyMatrix(g_diff),
    )
    # Ignore weights if unweighted
    assert ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_int, weights="unweighted"),
        ScipyAdjacencyMatrix(g_diff, weights="unweighted"),
    )
    assert not ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_int),
        ScipyAdjacencyMatrix(
            ss.coo_matrix(
                ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 0])),
            )  # change is here                                 ^^^
        ),
    )
    assert not ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_int),
        ScipyAdjacencyMatrix(
            ss.coo_matrix(
                ([1, 2, 0, 3, 3, 0], ([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 1, 2])),
            )  # extra element  ^^^                  ^^^                 ^^^
        ),
    )
    # weights don't match, so we take the fast path and declare them not equal
    assert not ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_int), ScipyAdjacencyMatrix(g_int, weights="any"),
    )
    # Node index affects comparison
    assert ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_int, node_index=IndexedNodes("ABC")),
        ScipyAdjacencyMatrix(
            ss.coo_matrix(([0, 3, 3, 2, 1], ([0, 0, 1, 2, 2], [0, 1, 0, 0, 2])),),
            node_index=IndexedNodes("BCA"),
        ),
    )
    # Transposed
    assert ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_int),
        ScipyAdjacencyMatrix(
            ss.coo_matrix(([1, 2, 0, 3, 3], ([0, 1, 1, 1, 2], [0, 0, 1, 2, 1])),),
            transposed=True,
        ),
    )
    assert ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_int, transposed=True),
        ScipyAdjacencyMatrix(
            ss.coo_matrix(([1, 2, 0, 3, 3], ([0, 1, 1, 1, 2], [0, 0, 1, 2, 1])),)
        ),
    )
    assert ScipyAdjacencyMatrix.Type.compare_objects(
        ScipyAdjacencyMatrix(g_int, transposed=True),
        ScipyAdjacencyMatrix(g_int, transposed=True),
    )
    with pytest.raises(TypeError):
        ScipyAdjacencyMatrix.Type.compare_objects(5, 5)
