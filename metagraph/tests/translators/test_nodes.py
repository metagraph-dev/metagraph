import pytest
from metagraph.plugins.python.types import PythonNodes
from metagraph.plugins.numpy.types import NumpyNodes, CompactNumpyNodes
from metagraph.plugins.graphblas.types import GrblasNodes
from metagraph import IndexedNodes, SequentialNodes
import numpy as np
import grblas


def test_python_2_compactnumpy(default_plugin_resolver):
    dpr = default_plugin_resolver
    rev_letters = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
    x = PythonNodes(
        {"A": 12.5, "B": 33.4, "Q": -1.2}, node_index=IndexedNodes(rev_letters)
    )
    assert x.num_nodes == 26
    # Convert python -> compactnumpy
    intermediate = CompactNumpyNodes(
        np.array([12.5, -1.2, 33.4]),
        {"A": 0, "B": 2, "Q": 1},
        node_index=IndexedNodes(rev_letters),
    )
    y = dpr.translate(x, CompactNumpyNodes)
    assert CompactNumpyNodes.Type.compare_objects(y, intermediate)
    # Convert python <- compactnumpy
    x2 = dpr.translate(y, PythonNodes)
    assert PythonNodes.Type.compare_objects(x, x2)


def test_compactnumpy_2_numpy(default_plugin_resolver):
    dpr = default_plugin_resolver
    rev_letters = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
    x = CompactNumpyNodes(
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
    intermediate = NumpyNodes(
        data, missing_mask=missing_mask, node_index=IndexedNodes(rev_letters)
    )
    y = dpr.translate(x, NumpyNodes)
    assert NumpyNodes.Type.compare_objects(y, intermediate)
    # Convert compactnumpy <- numpy
    x2 = dpr.translate(y, CompactNumpyNodes)
    assert CompactNumpyNodes.Type.compare_objects(x, x2)


def test_numpy_2_compactnumpy_dense(default_plugin_resolver):
    dpr = default_plugin_resolver
    data = np.array([1, 3, 5, 7, 9])
    x = NumpyNodes(data, node_index=SequentialNodes(5))
    assert x.num_nodes == 5
    # Convert numpy -> compactnumpy
    intermediate = CompactNumpyNodes(
        data, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, node_index=SequentialNodes(5)
    )
    y = dpr.translate(x, CompactNumpyNodes)
    assert CompactNumpyNodes.Type.compare_objects(y, intermediate)
    # Convert numpy <- compactnumpy
    x2 = dpr.translate(y, NumpyNodes)
    assert NumpyNodes.Type.compare_objects(x, x2)


def test_graphblas_python(default_plugin_resolver):
    dpr = default_plugin_resolver
    rev_letters = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
    x = GrblasNodes(
        grblas.Vector.from_values([9, 24, 25], [-1.2, 33.4, 12.5], size=26),
        node_index=IndexedNodes(rev_letters),
    )
    assert x.num_nodes == 26
    # Convert graphblas -> python
    intermediate = PythonNodes(
        {"A": 12.5, "B": 33.4, "Q": -1.2}, node_index=IndexedNodes(rev_letters)
    )
    y = dpr.translate(x, PythonNodes)
    assert PythonNodes.Type.compare_objects(y, intermediate)


def test_numpy_graphblas(default_plugin_resolver):
    dpr = default_plugin_resolver
    data = np.array([1, 1, 3, 1, 4, 1, -1])
    missing = data == 1
    x = NumpyNodes(data, missing_mask=missing, node_index=IndexedNodes("ABCDEFG"))
    assert x.num_nodes == 7
    # Convert numpy -> graphblas
    intermediate = GrblasNodes(
        grblas.Vector.from_values([2, 4, 6], [3, 4, -1]),
        node_index=IndexedNodes("ABCDEFG"),
    )
    y = dpr.translate(x, GrblasNodes)
    assert GrblasNodes.Type.compare_objects(y, intermediate)
