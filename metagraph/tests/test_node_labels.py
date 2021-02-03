import numpy as np
import pytest

from metagraph.core.node_labels import NodeLabels


def test_nodelabels():
    labels = NodeLabels([0, 10, 42], ["A", "B", "C"])
    assert labels == labels
    assert len(labels) == 3

    labels2 = NodeLabels([0, 10, 42], ["A", "C", "B"])
    assert labels != labels2

    labels3 = NodeLabels.from_dict({"A": 0, "C": 42, "B": 10})
    assert labels == labels3

    labels4 = NodeLabels.from_dict({0: "A", 42: "C", 10: "B"})
    assert labels == labels4

    # Cannot compare equality to dict
    assert labels != {"A": 0, "B": 10, "C": 42}
    assert {"A": 0, "B": 10, "C": 42} != labels


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


def test_nodelabels_errors():
    with pytest.raises(ValueError):
        NodeLabels([0, 1, 2], ["A", "B"])
    with pytest.raises(TypeError):
        NodeLabels(["A", "B", "C"], [0, 1, 2])
    with pytest.raises(TypeError, match="unhashable"):
        NodeLabels([0, 1, 2], [set(), dict(), list()])
    with pytest.raises(ValueError, match="duplicate node ids"):
        NodeLabels([0, 1, 1], ["A", "B", "C"])
    with pytest.raises(ValueError, match="duplicate labels"):
        NodeLabels([0, 1, 2], ["A", "C", "C"])
    with pytest.raises(TypeError, match="mapping must be dict-like"):
        NodeLabels.from_dict([("A", 1), ("B", 2)])
    with pytest.raises(ValueError, match="mapping is empty"):
        NodeLabels.from_dict({})
