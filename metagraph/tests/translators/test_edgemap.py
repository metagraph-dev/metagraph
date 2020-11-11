from metagraph.tests.util import default_plugin_resolver
from . import RoundTripper
from metagraph.plugins.pandas.types import PandasEdgeMap, PandasEdgeSet
import pandas as pd


def test_edgemap_roundtrip_directed(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df = pd.DataFrame(
        {
            "Source": [1, 3, 3, 5],
            "Target": [3, 1, 7, 5],
            "wgt_float": [1.1, 0, -3.3, 4.4],
            "wgt_int": [1, 0, -3, 4],
            "wgt_bool": [True, False, True, True],
        }
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_float", is_directed=True)
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_int", is_directed=True)
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_bool", is_directed=True)
    )


def test_edgemap_roundtrip_undirected(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df = pd.DataFrame(
        {
            "Source": [1, 5, 3, 5],
            "Target": [3, 3, 7, 5],
            "wgt_float": [1.1, 0, -3.3, 4.4],
            "wgt_int": [1, 0, -3, 4],
            "wgt_bool": [True, False, True, True],
        }
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_float", is_directed=False)
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_int", is_directed=False)
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_bool", is_directed=False)
    )


def test_edgemap_roundtrip_directed_symmetric(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df = pd.DataFrame(
        {
            "Source": [1, 3, 3, 5, 3, 7, 5],
            "Target": [3, 1, 5, 3, 7, 3, 5],
            "wgt_float": [1.1, 1.1, -3.3, -3.3, 0, 0, 2.0],
            "wgt_int": [1, 1, -3, -3, 0, 0, 2],
            "wgt_bool": [True, True, True, True, False, False, True],
        }
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_float", is_directed=True)
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_int", is_directed=True)
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_bool", is_directed=True)
    )


def test_edgemap_roundtrip_nonegweights(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df = pd.DataFrame(
        {
            "Source": [1, 3, 3, 5],
            "Target": [3, 1, 7, 5],
            "wgt_float": [1.1, 0, 3.3, 4.4],
            "wgt_int": [1, 0, 3, 4],
            "wgt_bool": [True, False, True, True],
        }
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_float", is_directed=True)
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_int", is_directed=True)
    )
    rt.verify_round_trip(
        PandasEdgeMap(df, "Source", "Target", "wgt_bool", is_directed=True)
    )


def test_edgemap_edgeset_oneway_directed(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df_start = pd.DataFrame(
        {"Source": [1, 3, 3, 5], "Target": [3, 1, 7, 5], "wgt": [1.1, 0, -3.3, 4.4]}
    )
    df_end = df_start[["Source", "Target"]].copy()
    rt.verify_one_way(
        PandasEdgeMap(df_start, "Source", "Target", "wgt", is_directed=True),
        PandasEdgeSet(df_end, "Source", "Target", is_directed=True),
    )


def test_edgemap_edgeset_oneway_undirected(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df_start = pd.DataFrame(
        {"Source": [1, 5, 3, 5], "Target": [3, 3, 7, 5], "wgt": [1.1, 0, -3.3, 4.4]}
    )
    df_end = df_start[["Source", "Target"]]
    rt.verify_one_way(
        PandasEdgeMap(df_start, "Source", "Target", "wgt", is_directed=False),
        PandasEdgeSet(df_end, "Source", "Target", is_directed=False),
    )


def test_edgemap_edgeset_oneway_directed_symmetric(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df_start = pd.DataFrame(
        {
            "Source": [1, 3, 3, 5, 3, 7, 5],
            "Target": [3, 1, 5, 3, 7, 3, 5],
            "wgt": [1.1, 1.1, -3.3, -3.3, 0, 0, 2.0],
        }
    )
    df_end = df_start[["Source", "Target"]]
    rt.verify_one_way(
        PandasEdgeMap(df_start, "Source", "Target", "wgt", is_directed=True),
        PandasEdgeSet(df_end, "Source", "Target", is_directed=True),
    )
