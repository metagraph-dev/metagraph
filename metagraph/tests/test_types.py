import pytest

from metagraph import types


def test_nodeset_not_implemented():
    nodeset = types.NodeSet()

    with pytest.raises(NotImplementedError):
        0 in nodeset

    with pytest.raises(NotImplementedError):
        len(nodeset)


def test_nodemap_not_implemented():
    nodemap = types.NodeMap()

    with pytest.raises(NotImplementedError):
        0 in nodemap

    with pytest.raises(NotImplementedError):
        nodemap[0]

    with pytest.raises(NotImplementedError):
        len(nodemap)


def test_node_id():
    assert str(types.NodeID) == "NodeID"

    with pytest.raises(NotImplementedError):
        types.NodeID(25)
