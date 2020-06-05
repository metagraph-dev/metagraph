import pytest

from metagraph import types


def test_nodemap_not_implemented():
    nodemap = types.NodeMap()

    with pytest.raises(NotImplementedError):
        nodemap[0]

    with pytest.raises(NotImplementedError):
        nodemap.num_nodes()
