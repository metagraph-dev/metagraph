import pytest

from metagraph import types


def test_nodemap_not_implemented():
    nodemap = types.NodeMap()

    with pytest.raises(NotImplementedError):
        nodemap[0]

    with pytest.raises(NotImplementedError):
        nodemap.num_nodes()

    with pytest.raises(NotImplementedError):
        nodemap.node_index()


def test_graph_not_implemented():
    graph = types.Graph()

    with pytest.raises(NotImplementedError):
        graph.num_nodes()

    with pytest.raises(NotImplementedError):
        graph.node_index()
