import numpy as np
import pytest

from metagraph.core.node_labels import NodeLabels


def test_nodelabels():
    with pytest.raises(ValueError):
        NodeLabels([0, 1, 2], ["A", "B"])
    with pytest.raises(TypeError):
        NodeLabels(["A", "B", "C"], [0, 1, 2])
    with pytest.raises(TypeError, match="unhashable"):
        NodeLabels([0, 1, 2], [set(), dict(), list()])

    labels = NodeLabels([0, 10, 42], ["A", "B", "C"])
    assert labels == labels

    labels2 = NodeLabels([0, 10, 42], ["A", "C", "B"])
    assert labels != labels2

    labels3 = NodeLabels.from_dict({"A": 0, "C": 42, "B": 10})
    assert labels == labels3


def test_labels_to_nodes():
    labels = NodeLabels(range(10), "abcdefghij")
    assert "j" in labels
    assert "q" not in labels
    assert labels["c"] == 2
    with pytest.raises(KeyError):
        labels["q"]
    with pytest.raises(KeyError):
        labels[("a", "g")]
    assert labels[["a", "g"]] == [0, 6]


def test_nodes_to_labels():
    labels = NodeLabels([0, 10, 42], ["A", "B", "C"])
    assert 10 in labels.ids
    assert 20 not in labels.ids
    assert labels.ids[10] == "B"
    with pytest.raises(KeyError):
        labels.ids[20]
    with pytest.raises(KeyError):
        labels.ids[(42, 0)]
    assert labels.ids[[42, 0]] == ["C", "A"]
