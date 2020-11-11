import pytest
from metagraph.tests.util import default_plugin_resolver
from . import RoundTripper
from metagraph.plugins.python.types import PythonNodeSetType
from metagraph.plugins.numpy.types import NumpyNodeSet, NumpyNodeMap
import numpy as np


def test_nodeset_roundtrip(default_plugin_resolver):
    rt = RoundTripper(default_plugin_resolver)
    ns = {2, 3, 55}
    rt.verify_round_trip(ns)


def test_np_nodemap_2_np_nodeset(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = NumpyNodeMap(np.array([00, 10, 20]))
    assert len(x) == 3
    intermediate = NumpyNodeSet(np.array([0, 1, 2]))
    y = dpr.translate(x, NumpyNodeSet)
    dpr.assert_equal(y, intermediate)


def test_np_nodeset_2_py_nodeset(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = NumpyNodeSet(np.array([9, 5, 1]))
    assert len(x) == 3
    intermediate = {5, 1, 9}
    y = dpr.translate(x, PythonNodeSetType)
    dpr.assert_equal(y, intermediate)


def test_py_nodeset_2_np_nodeset(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = {2, 1, 5}
    assert len(x) == 3
    intermediate = NumpyNodeSet.from_mask(
        np.array([False, True, True, False, False, True])
    )
    y = dpr.translate(x, NumpyNodeSet)
    dpr.assert_equal(y, intermediate)
