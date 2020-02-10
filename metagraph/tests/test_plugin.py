from metagraph.core import plugin
import pytest


class MyAbstractType(plugin.AbstractType):
    pass


def test_abstract_type():
    # all instances are identical
    assert MyAbstractType() == MyAbstractType()
    # is hashable
    assert hash(MyAbstractType())


class MySimpleType(plugin.ConcreteType):
    abstract = MyAbstractType
    value_class = int
    target = "pdp11"


class MyType(plugin.ConcreteType):
    abstract = MyAbstractType
    value_class = str
    allowed_props = dict(lowercase=bool)
    target = "pdp11"

    @classmethod
    def get_type(cls, obj):
        if isinstance(obj, cls.value_class):
            is_lower = obj.lower() == obj
            return cls(lowercase=is_lower)
        else:
            raise TypeError(f"object not of type f{cls.__class__}")


def test_concrete_type():
    ct = MyType()
    ct_lower = MyType(lowercase=True)

    # equality and hashing
    assert ct == ct
    assert ct != ct_lower
    assert hash(ct) == hash(ct)

    # allowed properties
    with pytest.raises(KeyError, match="not allowed property"):
        bad_ct = MyType(not_a_prop=4)

    # specialization
    assert ct.is_satisfied_by(ct_lower)
    assert not ct_lower.is_satisfied_by(ct)

    # default typeof
    assert MySimpleType.get_type(10) == MySimpleType()
    with pytest.raises(TypeError, match="not of type"):
        MySimpleType.get_type(set())

    # custom typeof
    assert ct.is_typeof("python")
    assert MyType.get_type("python") == ct_lower
    assert MyType.get_type("PYTHon") != ct_lower
    with pytest.raises(TypeError, match="not of type"):
        MyType.get_type(4)


def test_translator():
    @plugin.translator
    def int_to_str(src: MySimpleType) -> MyType:
        return str(src)

    assert isinstance(int_to_str, plugin.Translator)
    # FIXME: check type attributes
    assert int_to_str(4) == "4"


def test_abstract_algorithm():
    @plugin.abstract_algorithm("power")
    def power(
        x: MyAbstractType, p: MyAbstractType
    ) -> MyAbstractType:  # pragma: no cover
        pass

    assert isinstance(power, plugin.AbstractAlgorithm)
    assert power.name == "power"


def test_concrete_algorithm():
    @plugin.concrete_algorithm("power")
    def int_power(x: MySimpleType, p: MySimpleType) -> MySimpleType:
        return x ** p

    assert isinstance(int_power, plugin.ConcreteAlgorithm)
    assert int_power.abstract_name == "power"
    assert int_power(2, 3) == 8
