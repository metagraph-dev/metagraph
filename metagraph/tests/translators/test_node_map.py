import pytest

grblas = pytest.importorskip("grblas")

from metagraph import NodeLabels
from metagraph.tests.util import default_plugin_resolver
from metagraph.plugins.python.types import PythonNodeMap
from metagraph.plugins.numpy.types import NumpyNodeMap
from metagraph.plugins.graphblas.types import GrblasNodeMap
import numpy as np
import grblas


def test_python_2_numpy(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = PythonNodeMap({0: 12.5, 1: 33.4, 42: -1.2})
    assert x.num_nodes == 3
    # Convert python -> numpy
    intermediate = NumpyNodeMap(
        np.array([12.5, 33.4, -1.2]), node_ids={0: 0, 1: 1, 42: 2},
    )
    y = dpr.translate(x, NumpyNodeMap)
    dpr.assert_equal(y, intermediate)
    # Convert python <- numpy
    x2 = dpr.translate(y, PythonNodeMap)
    dpr.assert_equal(x, x2)


# TODO: revive these once self translator is written
# def test_compactnumpy_2_numpy(default_plugin_resolver):
#     dpr = default_plugin_resolver
#     x = CompactNumpyNodeMap(np.array([12.5, 33.4, -1.2]), {0: 0, 1: 1, 12: 2},)
#     assert x.num_nodes == 3
#     # Convert compactnumpy -> numpy
#     data = np.array([12.5, 33.4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1.2])
#     missing_mask = data == 1
#     intermediate = NumpyNodeMap(data, missing_mask=missing_mask)
#     y = dpr.translate(x, NumpyNodeMap)
#     dpr.assert_equal(y, intermediate)
#     # Convert compactnumpy <- numpy
#     x2 = dpr.translate(y, CompactNumpyNodeMap)
#     dpr.assert_equal(x, x2)


# def test_numpy_default_index_2_numpy(default_plugin_resolver):
#     dpr = default_plugin_resolver
#     x = NumpyNodeMap(np.array([11.1, 33.3, -22.2]), node_ids={0: 0, 1: 1, 2: 2},)
#     assert x.num_nodes == 3
#     # Convert numpy -> numpy
#     data = np.array([11.1, 33.3, -22.2])
#     intermediate = NumpyNodeMap(data)
#     y = dpr.translate(x, NumpyNodeMap)
#     dpr.assert_equal(y, intermediate)
#     # # Convert numpy <- numpy
#     x2 = dpr.translate(y, NumpyNodeMap)
#     dpr.assert_equal(x, x2)


# def test_numpy_2_compactnumpy_dense(default_plugin_resolver):
#     dpr = default_plugin_resolver
#     data = np.array([1, 3, 5, 7, 9])
#     x = NumpyNodeMap(data)
#     assert x.num_nodes == 5
#     # Convert numpy -> compactnumpy
#     intermediate = CompactNumpyNodeMap(data, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4})
#     y = dpr.translate(x, CompactNumpyNodeMap)
#     dpr.assert_equal(y, intermediate)
#     # Convert numpy <- compactnumpy
#     x2 = dpr.translate(y, NumpyNodeMap)
#     dpr.assert_equal(x, x2)


def test_graphblas_python(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = GrblasNodeMap(
        grblas.Vector.from_values([9, 24, 25], [-1.2, 33.4, 12.5], size=26),
    )
    assert x.num_nodes == 3
    # Convert graphblas -> python
    intermediate = PythonNodeMap({25: 12.5, 24: 33.4, 9: -1.2})
    y = dpr.translate(x, PythonNodeMap)
    dpr.assert_equal(y, intermediate)


def test_numpy_graphblas(default_plugin_resolver):
    dpr = default_plugin_resolver
    data = np.array([1, 1, 3, 1, 4, 1, -1])
    missing = data == 1
    x = NumpyNodeMap(data, mask=~missing)
    assert x.num_nodes == 3
    # Convert numpy -> graphblas
    intermediate = dpr.wrappers.NodeMap.GrblasNodeMap(
        grblas.Vector.from_values([2, 4, 6], [3, 4, -1]),
    )
    # NOTE: this tests DelayedWrappers in dask mode in addition to the normal translation
    y = dpr.translate(x, dpr.wrappers.NodeMap.GrblasNodeMap)
    dpr.assert_equal(y, intermediate)
