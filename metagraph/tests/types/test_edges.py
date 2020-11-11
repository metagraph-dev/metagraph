import pytest

grblas = pytest.importorskip("grblas")

from metagraph.plugins.pandas.types import PandasEdgeMap, PandasEdgeSet
from metagraph.plugins.graphblas.types import GrblasEdgeMap
from metagraph.plugins.scipy.types import ScipyEdgeMap
from metagraph import NodeLabels
import pandas as pd
import scipy.sparse as ss
import numpy as np


def test_pandas_edge():
    # A -> A (weight=1)
    # A -> B (weight=2)
    # B -> B (weight=0)
    # B -> C (weight=3)
    # C -> B (weight=3)
    iprops = {"is_directed": True, "dtype": "int"}
    fprops = {"is_directed": True, "dtype": "float"}
    df = pd.DataFrame(
        {
            "source": ["A", "A", "B", "B", "C"],
            "target": ["A", "B", "B", "C", "B"],
            "weight": [1, 2, 0, 3, 3],
        }
    )
    PandasEdgeMap.Type.assert_equal(
        PandasEdgeMap(df), PandasEdgeMap(df.copy()), iprops, iprops, {}, {}
    )
    df_float = df.copy()
    df_float["weight"] = df_float["weight"].astype(np.float64)
    df_close = df_float.copy()
    df_close.loc[0, "weight"] = 1.0000000000001
    PandasEdgeMap.Type.assert_equal(
        PandasEdgeMap(df_close), PandasEdgeMap(df_float), fprops, fprops, {}, {}
    )
    diff1 = df.copy()
    diff1.loc[4, "weight"] = 333
    with pytest.raises(AssertionError):
        PandasEdgeMap.Type.assert_equal(
            PandasEdgeMap(df), PandasEdgeMap(diff1), iprops, iprops, {}, {}
        )
    # Edgesets ignore weights
    PandasEdgeSet.Type.assert_equal(
        PandasEdgeSet(df),
        PandasEdgeSet(diff1),
        {"is_directed": True},
        {"is_directed": True},
        {},
        {},
    )
    diff2 = df.copy()
    diff2.loc[4, "target"] = "A"
    with pytest.raises(AssertionError):
        PandasEdgeMap.Type.assert_equal(
            PandasEdgeMap(df), PandasEdgeMap(diff2), iprops, iprops, {}, {}
        )
    extra = df.copy()
    extra = extra.append(pd.Series([2], index=["weight"], name=("C", "A")))
    with pytest.raises(AssertionError):
        PandasEdgeMap.Type.assert_equal(
            PandasEdgeMap(df), PandasEdgeMap(extra), iprops, iprops, {}, {}
        )
    # Undirected cannot have duplicates
    with pytest.raises(ValueError):
        PandasEdgeMap(df, is_directed=False)

    # Different weight_label
    wgt = df.copy()
    wgt = wgt.rename(columns={"weight": "WGT"})
    PandasEdgeMap.Type.assert_equal(
        PandasEdgeMap(df, weight_label="weight"),
        PandasEdgeMap(wgt, weight_label="WGT"),
        iprops,
        iprops,
        {},
        {},
    )


def test_graphblas():
    # [1 2  ]
    # [  0 3]
    # [  3  ]
    iprops = {"is_directed": True, "dtype": "int"}
    fprops = {"is_directed": True, "dtype": "float"}
    g_int = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3], dtype=grblas.dtypes.INT64
    )
    g_float = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3], dtype=grblas.dtypes.FP64
    )
    GrblasEdgeMap.Type.assert_equal(
        GrblasEdgeMap(g_int), GrblasEdgeMap(g_int.dup()), iprops, iprops, {}, {}
    )
    g_close = g_float.dup()
    g_close[0, 0] = 1.0000000000001
    GrblasEdgeMap.Type.assert_equal(
        GrblasEdgeMap(g_close), GrblasEdgeMap(g_float), fprops, fprops, {}, {}
    )
    g_diff = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 3, 0, 3, 3]
    )  # change is here                     ^^^
    with pytest.raises(AssertionError):
        GrblasEdgeMap.Type.assert_equal(
            GrblasEdgeMap(g_int), GrblasEdgeMap(g_diff), iprops, iprops, {}, {}
        )
    with pytest.raises(AssertionError):
        GrblasEdgeMap.Type.assert_equal(
            GrblasEdgeMap(g_int),
            GrblasEdgeMap(
                grblas.Matrix.from_values(
                    [0, 0, 1, 1, 2], [0, 1, 1, 2, 0], [1, 2, 0, 3, 3]
                )  # change is here              ^^^
            ),
            iprops,
            iprops,
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
            iprops,
            iprops,
            {},
            {},
        )


def test_scipy():
    # [1 2  ]
    # [  0 3]
    # [  3  ]
    iprops = {"is_directed": True, "dtype": "int"}
    fprops = {"is_directed": True, "dtype": "float"}
    g_int = ss.coo_matrix(
        ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1])), dtype=np.int64
    )
    g_float = ss.coo_matrix(
        ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1])), dtype=np.float64
    )
    ScipyEdgeMap.Type.assert_equal(
        ScipyEdgeMap(g_int), ScipyEdgeMap(g_int.copy().tocsr()), iprops, iprops, {}, {}
    )
    g_close = g_float.tocsr()
    g_close[0, 0] = 1.0000000000001
    ScipyEdgeMap.Type.assert_equal(
        ScipyEdgeMap(g_close), ScipyEdgeMap(g_float), fprops, fprops, {}, {}
    )
    g_diff = ss.coo_matrix(
        ([1, 3, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1]))
    )  # -  ^^^ changed
    with pytest.raises(AssertionError):
        ScipyEdgeMap.Type.assert_equal(
            ScipyEdgeMap(g_int), ScipyEdgeMap(g_diff), iprops, iprops, {}, {}
        )
    with pytest.raises(AssertionError):
        ScipyEdgeMap.Type.assert_equal(
            ScipyEdgeMap(g_int),
            ScipyEdgeMap(
                ss.coo_matrix(
                    ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 0]))
                )  # change is here                                 ^^^
            ),
            iprops,
            iprops,
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
            iprops,
            iprops,
            {},
            {},
        )
    # Node index affects comparison
    ScipyEdgeMap.Type.assert_equal(
        ScipyEdgeMap(g_int, [0, 2, 7]),
        ScipyEdgeMap(g_int, [0, 2, 7]),
        iprops,
        iprops,
        {},
        {},
    )
    #    0 2 7      2 7 0
    # 0 [1 2  ]  2 [0 3  ]
    # 2 [  0 3]  7 [3    ]
    # 7 [  3  ]  0 [2   1]
    g_int_reordered = ss.coo_matrix(
        ([0, 3, 3, 2, 1], ([0, 0, 1, 2, 2], [0, 1, 0, 0, 2])), dtype=np.int64
    )
    ScipyEdgeMap.Type.assert_equal(
        ScipyEdgeMap(g_int, [0, 2, 7]),
        ScipyEdgeMap(g_int_reordered, [2, 7, 0]),
        iprops,
        iprops,
        {},
        {},
    )
    with pytest.raises(AssertionError):
        ScipyEdgeMap.Type.assert_equal(
            ScipyEdgeMap(g_int, [0, 2, 7]),
            ScipyEdgeMap(g_int, [0, 1, 2]),
            iprops,
            iprops,
            {},
            {},
        )
