import os
import sys
import math
import pytest

from metagraph.core import plugin
from metagraph.core.resolver import Resolver


@pytest.fixture
def site_dir():
    test_site_dir = os.path.join(os.path.dirname(__file__), "site_dir")
    sys.path.insert(0, test_site_dir)
    yield test_site_dir
    sys.path.remove(test_site_dir)


class MyAbstractType(plugin.AbstractType):
    pass


class MyNumericAbstractType(plugin.AbstractType):
    properties = {"positivity": ["any", ">=0", ">0"], "divisible_by_two": [False, True]}


class IntType(plugin.ConcreteType, abstract=MyNumericAbstractType):
    value_type = int
    target = "pdp11"

    @classmethod
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, int):
            ret_val = cls()
            kwargs = {"positivity": "any", "divisible_by_two": obj % 2 == 0}
            if obj > 0:
                kwargs["positivity"] = ">0"
            elif obj == 0:
                kwargs["positivity"] = ">=0"
            ret_val.abstract_instance = MyNumericAbstractType(**kwargs)
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")


class FloatType(plugin.ConcreteType, abstract=MyNumericAbstractType):
    value_type = float
    target = "pdp11"

    @classmethod
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, float):
            ret_val = cls()
            positivity = "any"
            if obj > 0:
                positivity = ">0"
            elif obj == 0:
                positivity = ">=0"
            ret_val.abstract_instance = MyNumericAbstractType(positivity=positivity)
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")


class StrNum(plugin.Wrapper, abstract=MyNumericAbstractType):
    def __init__(self, val):
        self.value = val
        assert isinstance(val, str)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.value == other.value

    @classmethod
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, StrNum):
            value = obj.value
            ret_val = cls()
            positivity = ">0"
            if value.startswith("-"):
                positivity = "any"
            elif value == "0":
                positivity = ">=0"
            ret_val.abstract_instance = MyNumericAbstractType(positivity=positivity)
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")


class StrType(plugin.ConcreteType, abstract=MyAbstractType):
    value_type = str
    allowed_props = dict(lowercase=bool)
    target = "pdp11"

    @classmethod
    def get_type(cls, obj):
        if isinstance(obj, cls.value_type):
            is_lower = obj.lower() == obj
            return cls(lowercase=is_lower)
        else:
            raise TypeError(f"object not of type {cls.__class__}")


class OtherType(plugin.ConcreteType, abstract=MyAbstractType):
    target = "pdp11"


@plugin.translator
def int_to_str(src: IntType) -> StrNum:
    """Convert int to str"""
    return StrNum(str(src))


@plugin.translator
def str_to_int(src: StrNum) -> IntType:
    """Convert str to int"""
    return int(src.value)


@plugin.abstract_algorithm("power")
def abstract_power(
    x: MyNumericAbstractType, p: MyNumericAbstractType
) -> MyNumericAbstractType:  # pragma: no cover
    """Raise x to the power of p"""
    pass


@plugin.concrete_algorithm("power")
def int_power(x: IntType, p: IntType) -> IntType:
    return x ** p


@plugin.abstract_algorithm("ln")
def abstract_ln(
    x: MyNumericAbstractType(positivity=">0"),
) -> MyNumericAbstractType:  # pragma: no cover
    """Take the natural log"""
    pass


@plugin.concrete_algorithm("ln")
def float_ln(x: FloatType) -> FloatType:
    return math.log(x)


# Handy for manual testing
def make_example_resolver():
    res = Resolver()
    res.register(
        abstract_types={MyAbstractType, MyNumericAbstractType},
        concrete_types={StrType, IntType, FloatType, OtherType},
        wrappers={StrNum},
        translators={int_to_str, str_to_int},
        abstract_algorithms={abstract_power, abstract_ln},
        concrete_algorithms={int_power, float_ln},
    )
    return res


@pytest.fixture
def example_resolver():
    return make_example_resolver()
