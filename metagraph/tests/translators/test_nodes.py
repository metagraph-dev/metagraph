import pytest
from metagraph.tests.util import default_plugin_resolver
from metagraph.plugins.python.types import PythonNodeMap
from metagraph.plugins.numpy.types import NumpyNodeMap, CompactNumpyNodeMap
from metagraph.plugins.graphblas.types import GrblasNodeMap
from metagraph import IndexedNodes, SequentialNodes
import numpy as np
import grblas


def test_python_2_compactnumpy(default_plugin_resolver):
    dpr = default_plugin_resolver
    rev_letters = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
    x = PythonNodeMap(
        {"A": 12.5, "B": 33.4, "Q": -1.2}, node_index=IndexedNodes(rev_letters)
    )
    assert x.num_nodes == 26
    # Convert python -> compactnumpy
    intermediate = CompactNumpyNodeMap(
        np.array([12.5, -1.2, 33.4]),
        {"A": 0, "B": 2, "Q": 1},
        node_index=IndexedNodes(rev_letters),
    )
    y = dpr.translate(x, CompactNumpyNodeMap)
    CompactNumpyNodeMap.Type.assert_equal(y, intermediate)
    # Convert python <- compactnumpy
    x2 = dpr.translate(y, PythonNodeMap)
    PythonNodeMap.Type.assert_equal(x, x2)


def test_compactnumpy_2_numpy(default_plugin_resolver):
    dpr = default_plugin_resolver
    rev_letters = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
    x = CompactNumpyNodeMap(
        np.array([12.5, 33.4, -1.2]),
        {"A": 0, "B": 1, "Q": 2},
        node_index=IndexedNodes(rev_letters),
    )
    assert x.num_nodes == 26
    # Convert compactnumpy -> numpy
    # fmt: off
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1.2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 33.4, 12.5])
    # fmt: on
    missing_mask = data == 1
    intermediate = NumpyNodeMap(
        data, missing_mask=missing_mask, node_index=IndexedNodes(rev_letters)
    )
    y = dpr.translate(x, NumpyNodeMap)
    NumpyNodeMap.Type.assert_equal(y, intermediate)
    # Convert compactnumpy <- numpy
    x2 = dpr.translate(y, CompactNumpyNodeMap)
    CompactNumpyNodeMap.Type.assert_equal(x, x2)


def test_compactnumpy_default_index_2_numpy(default_plugin_resolver):
    dpr = default_plugin_resolver
    x = CompactNumpyNodeMap(np.array([11.1, 33.3, -22.2]), {"A": 0, "B": 2, "C": 1},)
    assert x.num_nodes == 3
    # Convert compactnumpy -> numpy
    data = np.array([11.1, 33.3, -22.2])
    intermediate = NumpyNodeMap(data, node_index=IndexedNodes("ACB"))
    y = dpr.translate(x, NumpyNodeMap)
    NumpyNodeMap.Type.assert_equal(y, intermediate)
    # # Convert compactnumpy <- numpy
    x2 = dpr.translate(y, CompactNumpyNodeMap)
    CompactNumpyNodeMap.Type.assert_equal(x, x2)


def test_numpy_2_compactnumpy_dense(default_plugin_resolver):
    dpr = default_plugin_resolver
    data = np.array([1, 3, 5, 7, 9])
    x = NumpyNodeMap(data, node_index=SequentialNodes(5))
    assert x.num_nodes == 5
    # Convert numpy -> compactnumpy
    intermediate = CompactNumpyNodeMap(
        data, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, node_index=SequentialNodes(5)
    )
    y = dpr.translate(x, CompactNumpyNodeMap)
    CompactNumpyNodeMap.Type.assert_equal(y, intermediate)
    # Convert numpy <- compactnumpy
    x2 = dpr.translate(y, NumpyNodeMap)
    NumpyNodeMap.Type.assert_equal(x, x2)


def test_graphblas_python(default_plugin_resolver):
    dpr = default_plugin_resolver
    rev_letters = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
    x = GrblasNodeMap(
        grblas.Vector.from_values([9, 24, 25], [-1.2, 33.4, 12.5], size=26),
        node_index=IndexedNodes(rev_letters),
    )
    assert x.num_nodes == 26
    # Convert graphblas -> python
    intermediate = PythonNodeMap(
        {"A": 12.5, "B": 33.4, "Q": -1.2}, node_index=IndexedNodes(rev_letters)
    )
    y = dpr.translate(x, PythonNodeMap)
    PythonNodeMap.Type.assert_equal(y, intermediate)


def test_numpy_graphblas(default_plugin_resolver):
    dpr = default_plugin_resolver
    data = np.array([1, 1, 3, 1, 4, 1, -1])
    missing = data == 1
    x = NumpyNodeMap(data, missing_mask=missing, node_index=IndexedNodes("ABCDEFG"))
    assert x.num_nodes == 7
    # Convert numpy -> graphblas
    intermediate = GrblasNodeMap(
        grblas.Vector.from_values([2, 4, 6], [3, 4, -1]),
        node_index=IndexedNodes("ABCDEFG"),
    )
    y = dpr.translate(x, GrblasNodeMap)
    GrblasNodeMap.Type.assert_equal(y, intermediate)


def test_numpy_rebuild_for_node_index(default_plugin_resolver):
    dpr = default_plugin_resolver
    data = np.array([1, 3, 5, 7, 9])
    nodes = NumpyNodeMap(data, node_index=SequentialNodes(5))
    nodes.rebuild_for_node_index(IndexedNodes({i: i for i in range(5)}))
    with pytest.raises(ValueError):
        nodes.rebuild_for_node_index(IndexedNodes({i + 1: i for i in range(5)}))
    with pytest.raises(ValueError):
        seq_nodes = SequentialNodes(3)
        seq_nodes._verify_valid_conversion(SequentialNodes(1))
