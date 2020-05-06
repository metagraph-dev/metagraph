import pytest

from metagraph import types


def test_nodes_not_implemented():
    nodes = types.Nodes()

    with pytest.raises(NotImplementedError):
        nodes[0]

    with pytest.raises(NotImplementedError):
        nodes.num_nodes()

    with pytest.raises(NotImplementedError):
        nodes.node_index()


def test_graph_not_implemented():
    graph = types.Graph()

    with pytest.raises(NotImplementedError):
        graph.num_nodes()

    with pytest.raises(NotImplementedError):
        graph.node_index()
