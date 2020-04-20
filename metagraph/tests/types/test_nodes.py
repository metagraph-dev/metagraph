import pytest
from metagraph.plugins.python.types import PythonNodes
from metagraph.plugins.numpy.types import NumpyNodes, CompactNumpyNodes
from metagraph.plugins.graphblas.types import GrblasNodes
from metagraph import IndexedNodes
import numpy as np
from grblas import Vector


def test_python():
    assert PythonNodes.Type.compare_objects(
        PythonNodes({"A": 1, "B": 2, "C": 3}), PythonNodes({"A": 1, "B": 2, "C": 3})
    )
    assert PythonNodes.Type.compare_objects(
        PythonNodes({"A": 1, "C": 3.333333333333333333333333, "B": 2}),
        PythonNodes({"A": 1, "C": 3.333333333333333333333334, "B": 2 + 1e-9}),
    )
    assert not PythonNodes.Type.compare_objects(
        PythonNodes({"A": 1}), PythonNodes({"A": 1, "B": 2})
    )
    assert not PythonNodes.Type.compare_objects(
        PythonNodes({"A": 1, "B": 22}), PythonNodes({"A": 1, "B": 2})
    )
    assert not PythonNodes.Type.compare_objects(
        PythonNodes({"A": 1.1}), PythonNodes({"A": 1})
    )
    # Node index doesn't affect equality
    assert PythonNodes.Type.compare_objects(
        PythonNodes({"A": 1, "B": 2, "C": 3}, node_index=IndexedNodes("ABC")),
        PythonNodes({"A": 1, "B": 2, "C": 3}, node_index=IndexedNodes("CBA")),
    )
    with pytest.raises(TypeError):
        PythonNodes.Type.compare_objects(5, 5)


def test_numpy():
    assert NumpyNodes.Type.compare_objects(
        NumpyNodes(np.array([1, 3, 5, 7, 9])), NumpyNodes(np.array([1, 3, 5, 7, 9]))
    )
    assert NumpyNodes.Type.compare_objects(
        NumpyNodes(np.array([1, 3, 5.5555555555555555555, 7, 9])),
        NumpyNodes(np.array([1, 3, 5.5555555555555555556, 7, 9 + 1e-9])),
    )
    assert not NumpyNodes.Type.compare_objects(
        NumpyNodes(np.array([1, 3, 5, 7, 9])), NumpyNodes(np.array([1, 3, 5, 7, 9, 11]))
    )
    # weights don't match, so we take the fast path and declare them not equal
    assert not NumpyNodes.Type.compare_objects(
        NumpyNodes(np.array([1, 3, 5, 7, 9])),
        NumpyNodes(np.array([1, 3, 5, 7, 9]), weights="any"),
    )
    # Missing value should not affect equality
    assert NumpyNodes.Type.compare_objects(
        NumpyNodes(
            np.array([1, -1, -1, 4, 5, -1]),
            missing_mask=np.array([False, True, True, False, False, True]),
        ),
        NumpyNodes(
            np.array([1, 0, 0, 4, 5, 0]),
            missing_mask=np.array([False, True, True, False, False, True]),
        ),
    )
    # Node index rearrangement
    assert NumpyNodes.Type.compare_objects(
        NumpyNodes(
            np.array([1, 4, 5, -1, 0, 1]),
            missing_mask=np.array([False, False, False, True, True, True]),
            node_index=IndexedNodes("ADEFBC"),
        ),
        NumpyNodes(
            np.array([1, -1, -1, 4, 5, -1]),
            missing_mask=np.array([False, True, True, False, False, True]),
            node_index=IndexedNodes("ABCDEF"),
        ),
    )
    with pytest.raises(TypeError):
        NumpyNodes.Type.compare_objects(5, 5)


def test_numpy_compact():
    assert CompactNumpyNodes.Type.compare_objects(
        CompactNumpyNodes(np.array([1, 3, 5]), {0: 0, 240: 1, 968: 2}),
        CompactNumpyNodes(np.array([1, 3, 5]), {0: 0, 240: 1, 968: 2}),
    )
    assert CompactNumpyNodes.Type.compare_objects(
        CompactNumpyNodes(np.array([1, 3, 5.5555555555555555555]), {0: 0, 1: 1, 2: 2}),
        CompactNumpyNodes(
            np.array([1, 3 + 1e-9, 5.5555555555555555556]), {0: 0, 1: 1, 2: 2}
        ),
    )
    assert not CompactNumpyNodes.Type.compare_objects(
        CompactNumpyNodes(np.array([1, 3, 5]), {0: 0, 1: 1, 2: 2}),
        CompactNumpyNodes(np.array([1, 3, 5, 7]), {0: 0, 1: 1, 2: 2, 3: 3}),
    )
    # weights don't match, so we take the fast path and declare them not equal
    assert not CompactNumpyNodes.Type.compare_objects(
        CompactNumpyNodes(np.array([1, 3, 5]), {0: 0, 1: 1, 2: 2}),
        CompactNumpyNodes(np.array([1, 3, 5]), {0: 0, 1: 1, 2: 2}, weights="any"),
    )
    # Storage reorder
    assert CompactNumpyNodes.Type.compare_objects(
        CompactNumpyNodes(np.array([1, 3, 5]), {"A": 0, "B": 1, "C": 2}),
        CompactNumpyNodes(np.array([5, 1, 3]), {"C": 0, "A": 1, "B": 2}),
    )
    # Node index doesn't affect equality
    assert CompactNumpyNodes.Type.compare_objects(
        CompactNumpyNodes(
            np.array([1, 3, 5]),
            {"A": 0, "B": 1, "C": 2},
            node_index=IndexedNodes("ADEFBC"),
        ),
        CompactNumpyNodes(
            np.array([1, 3, 5]),
            {"A": 0, "B": 1, "C": 2},
            node_index=IndexedNodes("ABCDEF"),
        ),
    )
    with pytest.raises(TypeError):
        CompactNumpyNodes.Type.compare_objects(5, 5)


def test_graphblas():
    assert GrblasNodes.Type.compare_objects(
        GrblasNodes(Vector.new_from_values([0, 1, 3, 4], [1, 2, 3, 4])),
        GrblasNodes(Vector.new_from_values([0, 1, 3, 4], [1, 2, 3, 4])),
    )
    assert GrblasNodes.Type.compare_objects(
        GrblasNodes(
            Vector.new_from_values([0, 1, 3, 4], [1.0, 2.0, 3.333333333333333333, 4.0])
        ),
        GrblasNodes(
            Vector.new_from_values(
                [0, 1, 3, 4], [1.0, 2.0, 3.333333333333333334, 4 + 1e-9]
            )
        ),
    )
    assert not GrblasNodes.Type.compare_objects(
        GrblasNodes(Vector.new_from_values([0, 1, 3, 4], [1, 2, 3, 4])),
        GrblasNodes(Vector.new_from_values([0, 1, 2, 4], [1, 2, 3, 4])),
    )
    assert not GrblasNodes.Type.compare_objects(
        GrblasNodes(Vector.new_from_values([0, 1, 2], [1, 2, 3])),
        GrblasNodes(Vector.new_from_values([0, 1, 2, 3], [1, 2, 3, 4])),
    )
    # weights don't match, so we take the fast path and declare them not equal
    assert not GrblasNodes.Type.compare_objects(
        GrblasNodes(Vector.new_from_values([0, 1], [1, 2])),
        GrblasNodes(Vector.new_from_values([0, 1], [1, 2]), weights="any"),
    )
    # Node index affects comparison
    assert GrblasNodes.Type.compare_objects(
        GrblasNodes(
            Vector.new_from_values([0, 1, 4], [1, 2, 3], size=5),
            node_index=IndexedNodes("ABCDE"),
        ),
        GrblasNodes(
            Vector.new_from_values([0, 2, 3], [2, 3, 1], size=5),
            node_index=IndexedNodes("BDEAC"),
        ),
    )
    with pytest.raises(TypeError):
        GrblasNodes.Type.compare_objects(5, 5)
