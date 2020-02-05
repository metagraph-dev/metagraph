import metagraph.plugin
import pytest


def test_abstract_type():
    a = metagraph.plugin.AbstractType(name="graph")
    assert a.name == "graph"


def test_concrete_type():
    c = metagraph.plugin.ConcreteType(abstract="graph", name="graph_networkx")
    assert c.abstract == "graph"
    assert c.name == "graph_networkx"


def test_translator():
    # basic attributes
    t = metagraph.plugin.Translator(srctype="graph_networkx", dsttype="graph_cugraph")
    assert t.srctype == "graph_networkx"
    assert t.dsttype == "graph_cugraph"

    # missing method
    with pytest.raises(NotImplementedError) as e:
        t.translate(1)
