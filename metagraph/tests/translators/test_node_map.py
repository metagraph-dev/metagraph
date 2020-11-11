import pytest

grblas = pytest.importorskip("grblas")

from metagraph import NodeLabels
from metagraph.tests.util import default_plugin_resolver
from . import RoundTripper
from metagraph.plugins.python.types import PythonNodeMapType
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyNodeSet
from metagraph.plugins.graphblas.types import GrblasNodeMap
import numpy as np
import grblas


def test_nodemap_roundtrip(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    nodes = np.array([1, 2, 42, 99])
    vals = np.array([12.5, 33.4, -1.2, 0.0])
    rt.verify_round_trip(NumpyNodeMap(vals, nodes=nodes))
    rt.verify_round_trip(NumpyNodeMap(vals.astype(int), nodes=nodes))
    rt.verify_round_trip(NumpyNodeMap(vals.astype(bool), nodes=nodes))


def test_nodemap_nodeset_oneway(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    nodes = np.array([1, 2, 42, 99])
    vals = np.array([12.5, 33.4, -1.2, 0.0])
    start = NumpyNodeMap(vals, nodes=nodes)
    end = NumpyNodeSet(nodes)
    rt.verify_one_way(start, end)


def test_python_2_numpy_node_ids(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = {0: 12.5, 1: 33.4, 42: -1.2}
    # Convert python -> numpy
    intermediate = NumpyNodeMap(np.array([12.5, 33.4, -1.2]), nodes=(0, 1, 42))
    y = dpr.translate(x, NumpyNodeMap)
    dpr.assert_equal(y, intermediate)
    # Convert python <- numpy
    x2 = dpr.translate(y, PythonNodeMapType)
    dpr.assert_equal(x, x2)


def test_graphblas_python(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = GrblasNodeMap(
        grblas.Vector.from_values([9, 24, 25], [-1.2, 33.4, 12.5], size=26),
    )
    assert len(x) == 3
    # Convert graphblas -> python
    intermediate = {25: 12.5, 24: 33.4, 9: -1.2}
    y = dpr.translate(x, PythonNodeMapType)
    dpr.assert_equal(y, intermediate)


def test_numpy_graphblas(default_plugin_resolver):
    dpr = default_plugin_resolver
    data = np.array([3, -1, 4])
    x = NumpyNodeMap(data, [2, 6, 4])
    assert len(x) == 3
    # Convert numpy -> graphblas
    intermediate = dpr.wrappers.NodeMap.GrblasNodeMap(
        grblas.Vector.from_values([2, 4, 6], [3, 4, -1]),
    )
    # NOTE: this tests DelayedWrappers in dask mode in addition to the normal translation
    y = dpr.translate(x, dpr.wrappers.NodeMap.GrblasNodeMap)
    dpr.assert_equal(y, intermediate)
