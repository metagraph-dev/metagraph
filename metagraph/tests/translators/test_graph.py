import pytest
from metagraph.plugins.scipy.types import ScipyAdjacencyMatrix
from metagraph.plugins.networkx.types import NetworkXGraph
from metagraph.plugins.graphblas.types import GrblasAdjacencyMatrix
from metagraph.plugins.pandas.types import PandasEdgeList


def test_networkx_scipy(default_plugin_resolver):
    dpr = default_plugin_resolver
    pytest.xfail("not written")


def test_scipy_graphblas(default_plugin_resolver):
    dpr = default_plugin_resolver
    pytest.xfail("not written")


def test_networkx_2_pandas(default_plugin_resolver):
    dpr = default_plugin_resolver
    pytest.xfail("not written")
