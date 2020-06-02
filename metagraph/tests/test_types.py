import pytest

from metagraph import types


def test_nodemap_not_implemented():
    nodemap = types.NodeMap()

    with pytest.raises(NotImplementedError):
        nodemap[0]

    with pytest.raises(NotImplementedError):
        nodemap.num_nodes()

    with pytest.raises(NotImplementedError):
        nodemap.to_nodeset()


def test_edgemap_not_implemented():
    graph = types.EdgeMap()

    with pytest.raises(NotImplementedError):
        graph.to_edgeset()
