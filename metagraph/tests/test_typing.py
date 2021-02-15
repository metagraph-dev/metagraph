import pytest
import metagraph as mg
from metagraph.core import typing as mgtyping
from metagraph.plugins.numpy.types import NumpyNodeMap


def test_node_id():
    assert str(mg.NodeID) == "NodeID"

    with pytest.raises(NotImplementedError):
        mg.NodeID(25)


def test_union():
    a = mg.Union[int, float]
    assert isinstance(a, mgtyping.Combo)
    assert not a.optional
    assert a.kind == "Union"
    assert len(a) == 2
    assert repr(a) == "mg.Union[<class 'int'>,<class 'float'>]"

    b = mg.Union[int, None]
    assert isinstance(b, mgtyping.Combo)
    assert b.optional
    assert a.kind == "Union"
    assert len(b) == 1

    with pytest.raises(TypeError, match="Union requires more than one parameter"):
        mg.Union[int]

    with pytest.raises(TypeError, match="Union requires more than one parameter"):
        mg.Union[(int,)]


def test_list():
    a = mg.List[int]
    assert isinstance(a, mgtyping.Combo)
    assert a.kind == "List"
    assert len(a) == 1
    assert repr(a) == "mg.List[<class 'int'>]"

    b = mg.List[NumpyNodeMap]
    assert isinstance(b, mgtyping.Combo)
    assert b.types[0] is NumpyNodeMap

    c = mg.List[(int,)]
    assert isinstance(c, mgtyping.Combo)
    assert c.types[0] is int

    with pytest.raises(TypeError, match="Too many parameters"):
        mg.List[int, float]


def test_optional():
    a = mg.Optional[int]
    assert isinstance(a, mgtyping.Combo)
    assert a.optional
    assert a.kind is None
    assert len(a) == 1
    assert repr(a) == "mg.Optional[<class 'int'>]"

    b = mg.Optional[mg.Union[int, float]]
    assert isinstance(b, mgtyping.Combo)
    assert b.optional
    assert b.kind == "Union"
    assert len(b) == 2

    c = mg.Optional[mg.List[int]]
    assert isinstance(c, mgtyping.Combo)
    assert c.optional
    assert c.kind == "List"
    assert len(c) == 1
