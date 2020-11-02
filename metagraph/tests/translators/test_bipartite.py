import pytest
from metagraph.tests.util import default_plugin_resolver
from . import RoundTripper


def test_graph_roundtrip(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    pytest.xfail()
