from metagraph.core import plugin
import pytest

from .util import (
    MyAbstractType,
    MyNumericAbstractType,
    StrType,
    IntType,
    FloatType,
    StrNum,
    int_to_str,
    abstract_power,
    int_power,
    abstract_ln,
    float_ln,
)


def test_abstract_type():
    # all instances are identical
    assert MyAbstractType() == MyAbstractType()
    # is hashable
    assert hash(MyAbstractType())

    # invalid properties
    with pytest.raises(KeyError, match="not a valid property"):
        MyAbstractType(foo=17)

    # abstract with properties checks
    assert MyNumericAbstractType(divisible_by_two=True) == MyNumericAbstractType(
        divisible_by_two=True
    )
    assert MyNumericAbstractType(divisible_by_two=True) != MyNumericAbstractType(
        divisible_by_two=False
    )
    assert hash(MyNumericAbstractType(divisible_by_two=False, positivity=">=0"))

    # property values
    at = MyNumericAbstractType(positivity=">=0")
    assert at.prop_val == {"positivity": ">=0", "divisible_by_two": None}

    class AbstractType1(plugin.AbstractType):
        properties = {"k1": ["v1", "v2", "v3"]}

    with pytest.raises(
        ValueError, match="Invalid setting for .+ property: .+; must be one of .+"
    ):

        class ConcreteType1(
            plugin.ConcreteType, abstract=AbstractType1(k1="bad_k1_value")
        ):
            pass  # pragma: no cover


def test_concrete_type():
    ct = StrType()
    ct_lower = StrType(lowercase=True)
    assert ct_lower["lowercase"]

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

    # default get_type
    assert IntType.get_type(10) == IntType()
    with pytest.raises(TypeError, match="not of type"):
        IntType.get_type(set())

    # custom typeclass_of
    assert ct.is_typeclass_of("python")
    assert not ct.is_typeclass_of(set())
    assert StrType.get_type("python") == ct_lower
    assert StrType.get_type("PYTHon") != ct_lower
    with pytest.raises(TypeError, match="not of type"):
        StrType.get_type(4)

    # compute abstract properties
    assert IntType.compute_abstract_properties(10, {"positivity"})["positivity"] == ">0"
    assert IntType.compute_abstract_properties(10, {"divisible_by_two"})[
        "divisible_by_two"
    ]
    with pytest.raises(KeyError, match="is not an abstract property of"):
        assert IntType.compute_abstract_properties(10, {"bad_abstract_property_label"})
    with pytest.raises(KeyError, match="is not an abstract property of"):
        StrType.compute_abstract_properties("arbitrary string", {"made_up_property"})

    # compute concrete properties
    assert StrType.compute_concrete_properties("lowercasestring", {"lowercase"})[
        "lowercase"
    ]
    with pytest.raises(KeyError, match="is not a concrete property of"):
        assert StrType.compute_concrete_properties("arbitrary string", ["bad_key"])
    with pytest.raises(KeyError, match="is not a concrete property of"):
        IntType.compute_concrete_properties(10, ["bad_key"])


def test_concrete_type_abstract_errors():
    with pytest.raises(TypeError, match="Missing required 'abstract' keyword argument"):

        class MyBadType(plugin.ConcreteType):
            pass

    with pytest.raises(TypeError, match="must be subclass of AbstractType"):

        class MyBadType(plugin.ConcreteType, abstract=4):
            pass


def test_wrapper():
    class StrNumZeroOnly(plugin.Wrapper, abstract=MyNumericAbstractType):
        def __init__(self, val):
            self.value = val
            self._assert_instance(val, str, f"{repr(val)} is not a string.")
            self._assert(0 == float(val), "'zero' is only valid value.")
            assert isinstance(val, str)

        class TypeMixin:
            pass

    assert StrNumZeroOnly("0")
    with pytest.raises(TypeError, match="'zero' is only valid value."):
        StrNumZeroOnly("1234")
    with pytest.raises(TypeError, match="is not a string."):
        StrNumZeroOnly(0)

    class StrNumZeroOnlyDefaultErrorMessage(
        plugin.Wrapper, abstract=MyNumericAbstractType
    ):
        def __init__(self, val):
            super().__init__()
            self._assert_instance(val, str)
            self._assert(0 == float(val), "str '0' is the only valid value.")

        class TypeMixin:
            pass

    # This should work
    StrNumZeroOnlyDefaultErrorMessage("0")
    # This will fail
    with pytest.raises(TypeError, match="is not an instance of"):
        StrNumZeroOnlyDefaultErrorMessage(0)


def test_translator():
    assert isinstance(int_to_str, plugin.Translator)
    assert int_to_str.__name__ == "int_to_str"
    assert "Convert int to str" in int_to_str.__doc__
    assert int_to_str(4).value == "4"


def test_abstract_algorithm():
    assert isinstance(abstract_power, plugin.AbstractAlgorithm)
    assert abstract_power.__name__ == "abstract_power"
    assert "Raise x to " in abstract_power.__doc__
    assert abstract_power.name == "power"


def test_concrete_algorithm():
    assert isinstance(int_power, plugin.ConcreteAlgorithm)
    assert int_power.abstract_name == "power"
    assert int_power(2, 3) == 8


def test_abstract_algorithm_with_properties():
    assert isinstance(abstract_ln, plugin.AbstractAlgorithm)
    assert abstract_ln.__name__ == "abstract_ln"
    assert abstract_ln.name == "ln"


def test_concrete_algorithm_with_properties():
    assert isinstance(float_ln, plugin.ConcreteAlgorithm)
    assert float_ln.abstract_name == "ln"
    assert abs(float_ln(100.0) - 4.605170185988092) < 1e-6
