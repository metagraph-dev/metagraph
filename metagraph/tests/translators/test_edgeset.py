from metagraph.tests.util import default_plugin_resolver
from . import RoundTripper
from metagraph.plugins.pandas.types import PandasEdgeSet
import pandas as pd


def test_edgeset_roundtrip_directed(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df = pd.DataFrame({"Source": [1, 3, 3, 5], "Target": [3, 1, 7, 5]})
    rt.verify_round_trip(PandasEdgeSet(df, "Source", "Target", is_directed=True))


def test_edgeset_roundtrip_undirected(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df = pd.DataFrame({"Source": [1, 5, 3, 5], "Target": [3, 3, 7, 5]})
    rt.verify_round_trip(PandasEdgeSet(df, "Source", "Target", is_directed=False))


def test_edgeset_roundtrip_directed_symmetric(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df = pd.DataFrame({"Source": [1, 3, 3, 5], "Target": [3, 1, 5, 3]})
    rt.verify_round_trip(PandasEdgeSet(df, "Source", "Target", is_directed=True))
