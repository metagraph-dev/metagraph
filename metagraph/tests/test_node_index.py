import numpy as np
import pytest

from metagraph.core.node_index import IndexedNodes, SequentialNodes


def test_repr_methods():
    assert isinstance(repr(IndexedNodes(np.array([1, 3, 5]))), str)
    assert isinstance(repr(SequentialNodes(5)), str)


def test_indexed_nodes():
    with pytest.raises(TypeError):
        IndexedNodes(np.array([[1, 2,], [3, 4]]))
    indexed_nodes_str = IndexedNodes("ABCDEFG")
    assert indexed_nodes_str != IndexedNodes("ABC")
    indexed_nodes_np = IndexedNodes(np.array([1, 3, 5]))
    assert indexed_nodes_np == IndexedNodes(np.array([1, 3, 5]))
    assert indexed_nodes_np != IndexedNodes([6, 7, 8, 9])


def test_sequential_nodes():
    seq_nodes = SequentialNodes(5)
    for i in range(5):
        assert seq_nodes.bylabel(i) == i
        assert seq_nodes.byindex(i) == i
    with pytest.raises(KeyError):
        seq_nodes.bylabel(9999)
    with pytest.raises(KeyError):
        seq_nodes.byindex(9999)
    assert seq_nodes.labels() == {0, 1, 2, 3, 4}
