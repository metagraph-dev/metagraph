import pytest
from metagraph.tests.util import default_plugin_resolver
from . import RoundTripper
import pandas as pd


def test_dataframe(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    df = pd.DataFrame(
        {
            "Source": [0, 1, 1, 3],
            "Target": [1, 2, 3, 1],
            "Weight": [44.2, 100.5, 9.7, 1.2],
        }
    )
    rt.verify_round_trip(df)
