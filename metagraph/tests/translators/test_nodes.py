import pytest
from metagraph.plugins.python.types import PythonNodes
from metagraph.plugins.numpy.types import NumpyNodes, CompactNumpyNodes
from metagraph.plugins.graphblas.types import GrblasNodes


def test_python_2_numpy(default_plugin_resolver):
    pytest.xfail("not written")


def test_python_graphblas(default_plugin_resolver):
    pytest.xfail("not written")


def test_graphblas_numpy(default_plugin_resolver):
    pytest.xfail("not written")
