import pytest

grblas = pytest.importorskip("grblas")

import metagraph as mg
from metagraph import NodeLabels
from metagraph.tests.util import default_plugin_resolver
from . import RoundTripper
from metagraph.plugins.python.types import PythonNodeMapType, PythonNodeSetType
from metagraph.plugins.numpy.types import NumpyNodeMap, NumpyNodeSet
from metagraph.plugins.graphblas.types import GrblasNodeMap, GrblasNodeSet
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


def test_method_call(default_plugin_resolver):
    dpr = default_plugin_resolver
    with dpr:
        x = NumpyNodeMap(np.array([12.5, 33.4, -1.2]), nodes=(0, 1, 42))
        y = {0: 12.5, 1: 33.4, 42: -1.2}
        z = GrblasNodeMap(grblas.Vector.from_values([0, 1, 42], [12.5, 33.4, -1.2]))
        z_other = GrblasNodeMap(grblas.Vector.from_values([0, 2], [1, 2]))
        # Wrapper
        dpr.assert_equal(x.translate(GrblasNodeMap), z)
        dpr.assert_equal(x.translate(dpr.wrappers.NodeMap.GrblasNodeMap), z)
        dpr.assert_equal(x.translate(mg.wrappers.NodeMap.GrblasNodeMap), z)
        # ConcreteType
        dpr.assert_equal(x.translate(GrblasNodeMap.Type), z)
        dpr.assert_equal(x.translate(PythonNodeMapType), y)
        dpr.assert_equal(x.translate(dpr.types.NodeMap.GrblasNodeMapType), z)
        dpr.assert_equal(x.translate(mg.types.NodeMap.GrblasNodeMapType), z)
        dpr.assert_equal(x.translate(mg.types.NodeMap.PythonNodeMapType), y)
        # ConcreteType's value_type
        dpr.assert_equal(x.translate(dict), y)
        # instance of Wrapper
        dpr.assert_equal(x.translate(z_other), z)
        # instance of ConcreteType's value_type
        dpr.assert_equal(x.translate({"a": 1, "b": 2}), y)
        # string of Wrapper class name
        dpr.assert_equal(x.translate("GrblasNodeMap"), z)
        # string of ConcreteType class name
        dpr.assert_equal(x.translate("PythonNodeMapType"), y)
        dpr.assert_equal(x.translate("GrblasNodeMapType"), z)
        dpr.assert_equal(x.translate("GrblasNodeMap.Type"), z)


def test_method_call_secondary(default_plugin_resolver):
    dpr = default_plugin_resolver
    with dpr:
        x = NumpyNodeMap(np.array([12.5, 33.4, -1.2]), nodes=(0, 1, 42))
        y = {0, 1, 42}
        z = GrblasNodeSet(grblas.Vector.from_values([0, 1, 42], [1, 1, 1]))
        z_other = GrblasNodeSet(grblas.Vector.from_values([0, 2], [1, 1]))
        # Wrapper
        dpr.assert_equal(x.translate(GrblasNodeSet), z)
        dpr.assert_equal(x.translate(dpr.wrappers.NodeSet.GrblasNodeSet), z)
        dpr.assert_equal(x.translate(mg.wrappers.NodeSet.GrblasNodeSet), z)
        # ConcreteType
        dpr.assert_equal(x.translate(GrblasNodeSet.Type), z)
        dpr.assert_equal(x.translate(PythonNodeSetType), y)
        dpr.assert_equal(x.translate(dpr.types.NodeSet.GrblasNodeSetType), z)
        dpr.assert_equal(x.translate(mg.types.NodeSet.GrblasNodeSetType), z)
        dpr.assert_equal(x.translate(mg.types.NodeSet.PythonNodeSetType), y)
        # ConcreteType's value_type
        dpr.assert_equal(x.translate(set), y)
        # instance of Wrapper
        dpr.assert_equal(x.translate(z_other), z)
        # instance of ConcreteType's value_type
        dpr.assert_equal(x.translate({"a", "b"}), y)
        # string of Wrapper class name
        dpr.assert_equal(x.translate("GrblasNodeSet"), z)
        # string of ConcreteType class name
        dpr.assert_equal(x.translate("PythonNodeSetType"), y)
        dpr.assert_equal(x.translate("GrblasNodeSetType"), z)
