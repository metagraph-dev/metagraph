from metagraph.core import plugin
import pytest

from .util import (
    MyAbstractType,
    StrType,
    IntType,
    int_to_str,
    abstract_power,
    int_power,
)


def test_abstract_type():
    # all instances are identical
    assert MyAbstractType() == MyAbstractType()
    # is hashable
    assert hash(MyAbstractType())


def test_concrete_type():
    ct = StrType()
    ct_lower = StrType(lowercase=True)

    # equality and hashing
    assert ct == ct
    assert ct != ct_lower
    assert hash(ct) == hash(ct)

    # allowed properties
    with pytest.raises(KeyError, match="not allowed property"):
        bad_ct = StrType(not_a_prop=4)

    # specialization
    assert ct.is_satisfied_by(ct_lower)
    assert not ct_lower.is_satisfied_by(ct)
    assert ct.is_satisfied_by_value("Python")
    assert ct.is_satisfied_by_value("python")
    assert not ct.is_satisfied_by_value(1)
    assert ct_lower.is_satisfied_by_value("python")
    assert not ct_lower.is_satisfied_by_value("Python")

    # default typeof
    assert IntType.get_type(10) == IntType()
    with pytest.raises(TypeError, match="not of type"):
        IntType.get_type(set())

    # custom typeof
    assert ct.is_typeof("python")
    assert not ct.is_typeof(set())
    assert StrType.get_type("python") == ct_lower
    assert StrType.get_type("PYTHon") != ct_lower
    with pytest.raises(TypeError, match="not of type"):
        StrType.get_type(4)


def test_translator():
    assert isinstance(int_to_str, plugin.Translator)
    # FIXME: check type attributes
    assert int_to_str(4) == "4"


def test_abstract_algorithm():
    assert isinstance(abstract_power, plugin.AbstractAlgorithm)
    assert abstract_power.name == "power"


def test_concrete_algorithm():
    assert isinstance(int_power, plugin.ConcreteAlgorithm)
    assert int_power.abstract_name == "power"
    assert int_power(2, 3) == 8
