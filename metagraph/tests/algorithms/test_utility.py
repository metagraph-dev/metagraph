from metagraph.tests.util import default_plugin_resolver
import numpy as np
from . import MultiVerify


def test_nodeset_choose_random(default_plugin_resolver):
    dpr = default_plugin_resolver
    py_node_set_unwrapped = {100, 200, 300, 400, 500, 600, 700}
    py_node_set = dpr.wrappers.NodeSet.PythonNodeSet(py_node_set_unwrapped)
    k = 3

    def cmp_func(x):
        assert x.num_nodes == k
        assert x.num_nodes < len(py_node_set_unwrapped)
        assert x.value.issubset(py_node_set_unwrapped)

    MultiVerify(dpr, "util.nodeset.choose_random", py_node_set, k).custom_compare(
        cmp_func, dpr.types.NodeSet.PythonNodeSetType
    )


def test_nodeset_sort(default_plugin_resolver):
    dpr = default_plugin_resolver
    py_node_map_unwrapped = {index: index * 100 for index in range(1, 8)}
    py_node_map = dpr.wrappers.NodeMap.PythonNodeMap(py_node_map_unwrapped)
    MultiVerify(dpr, "util.nodemap.sort", py_node_map).assert_equals(
        dpr.wrappers.Vector.NumpyVector(np.array([1, 2, 3, 4, 5, 6, 7]))
    )
    MultiVerify(dpr, "util.nodemap.sort", py_node_map, True, 4).assert_equals(
        dpr.wrappers.Vector.NumpyVector(np.array([1, 2, 3, 4]))
    )
    MultiVerify(dpr, "util.nodemap.sort", py_node_map, True).assert_equals(
        dpr.wrappers.Vector.NumpyVector(np.array([1, 2, 3, 4, 5, 6, 7]))
    )
    MultiVerify(dpr, "util.nodemap.sort", py_node_map, False, 3).assert_equals(
        dpr.wrappers.Vector.NumpyVector(np.array([7, 6, 5]))
    )
    MultiVerify(dpr, "util.nodemap.sort", py_node_map, False).assert_equals(
        dpr.wrappers.Vector.NumpyVector(np.array([7, 6, 5, 4, 3, 2, 1]))
    )


def test_nodemap_select(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = dpr.wrappers.NodeMap.PythonNodeMap({1: 11, 2: 22, 3: 33, 4: 44})
    node_set = dpr.wrappers.NodeSet.PythonNodeSet({2, 3})
    correct_answer = dpr.wrappers.NodeMap.PythonNodeMap({2: 22, 3: 33})
    MultiVerify(dpr, "util.nodemap.select", node_map, node_set).assert_equals(
        correct_answer
    )


def test_nodemap_filter(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = dpr.wrappers.NodeMap.PythonNodeMap({1: 11, 2: 22, 3: 33, 4: 44})
    correct_answer = dpr.wrappers.NodeSet.PythonNodeSet({2, 4})

    def filter_func(x):
        return x % 2 == 0

    MultiVerify(dpr, "util.nodemap.filter", node_map, filter_func).assert_equals(
        correct_answer
    )
    filter_func = lambda x: x % 2 == 0
    MultiVerify(dpr, "util.nodemap.filter", node_map, filter_func).assert_equals(
        correct_answer
    )


def test_nodemap_apply(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = dpr.wrappers.NodeMap.PythonNodeMap({1: 11, 2: 22, 3: 33, 4: 44})
    apply_func = lambda x: x * 100
    correct_answer = dpr.wrappers.NodeMap.PythonNodeMap(
        {1: 1100, 2: 2200, 3: 3300, 4: 4400}
    )
    MultiVerify(dpr, "util.nodemap.apply", node_map, apply_func).assert_equals(
        correct_answer
    )


def test_nodemap_reduce(default_plugin_resolver):
    dpr = default_plugin_resolver
    node_map = dpr.wrappers.NodeMap.PythonNodeMap({1: 11, 2: 22, 3: 33, 4: 44})
    reduce_func = lambda x, y: x + y
    correct_answer = 110
    MultiVerify(dpr, "util.nodemap.reduce", node_map, reduce_func).assert_equals(
        correct_answer
    )
