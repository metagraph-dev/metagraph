import pytest
from metagraph.tests.util import default_plugin_resolver
from metagraph.plugins.python.types import PythonNodeSet
from metagraph.plugins.numpy.types import NumpyNodeSet, NumpyNodeMap
import numpy as np


def test_np_nodemap_2_np_nodeset(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = NumpyNodeMap(np.array([00, 10, 20]))
    assert x.num_nodes == 3
    intermediate = NumpyNodeSet(np.array([0, 1, 2]))
    y = dpr.translate(x, NumpyNodeSet)
    dpr.assert_equal(y, intermediate)


def test_np_nodeset_2_py_nodeset(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = NumpyNodeSet(np.array([1, 5, 9]))
    assert x.num_nodes == 3
    intermediate = PythonNodeSet({5, 1, 9})
    y = dpr.translate(x, PythonNodeSet)
    dpr.assert_equal(y, intermediate)


def test_py_nodeset_2_np_nodeset(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = PythonNodeSet({5, 1, 9})
    assert x.num_nodes == 3
    intermediate = NumpyNodeSet(np.array([1, 5, 9]))
    y = dpr.translate(x, NumpyNodeSet)
    dpr.assert_equal(y, intermediate)
