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
    assert len(a) == 2

    b = mg.Union[int, None]
    assert isinstance(b, mgtyping.Combo)
    assert b.optional
    assert len(b) == 1

    with pytest.raises(TypeError, match="Union requires more than one parameter"):
        mg.Union[int]

    with pytest.raises(TypeError, match="Union requires more than one parameter"):
        mg.Union[(int,)]


def test_optional():
    a = mg.Optional[mg.Union[int, float]]
    assert isinstance(a, mgtyping.Combo)
    assert a.optional
    assert len(a) == 2

    b = mg.Optional[float]
    assert isinstance(b, mgtyping.Combo)
    assert b.optional
    assert len(b) == 1


def test_combo():
    with pytest.raises(TypeError, match="type within Union or Optional may not be"):
        mg.Optional[7]

    with pytest.raises(TypeError, match="type within Union or Optional may not be"):
        mg.Union[7, 14]

    with pytest.raises(TypeError, match="Must be optional if only one type"):
        mgtyping.Combo([int], optional=False)

    with pytest.raises(
        TypeError, match="Strict is required for multiple allowable types"
    ):
        mgtyping.Combo([int, float], strict=False)


def test_uniform_iterable():
    a = mg.List[int]
    assert isinstance(a, mgtyping.UniformIterable)
    assert a.container_name == "List"
    assert repr(a) == "List[<class 'int'>]"

    b = mg.List[NumpyNodeMap]
    assert isinstance(b, mgtyping.UniformIterable)
    assert isinstance(b.element_type, NumpyNodeMap.Type)

    c = mg.List[(int,)]
    assert isinstance(c, mgtyping.UniformIterable)
    assert c.element_type is int

    with pytest.raises(TypeError, match="Too many parameters"):
        mg.List[int, float]
