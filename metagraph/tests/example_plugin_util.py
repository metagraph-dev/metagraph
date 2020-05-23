import math
from typing import List, Dict, Any

from metagraph.core import plugin


class MyAbstractType(plugin.AbstractType):
    pass


class MyNumericAbstractType(plugin.AbstractType):
    properties = {"positivity": ["any", ">=0", ">0"], "divisible_by_two": [False, True]}


class IntType(plugin.ConcreteType, abstract=MyNumericAbstractType):
    value_type = int
    target = "pdp11"

    @classmethod
    def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
        cls._validate_abstract_props(props)
        # return all properties regardless of what was requested, as
        # is permitted by the interface
        ret = {"positivity": "any", "divisible_by_two": obj % 2 == 0}
        if obj > 0:
            ret["positivity"] = ">0"
        elif obj == 0:
            ret["positivity"] = ">=0"

        return ret


class FloatType(plugin.ConcreteType, abstract=MyNumericAbstractType):
    value_type = float
    target = "pdp11"

    @classmethod
    def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
        cls._validate_abstract_props(props)
        # return all properties regardless of what was requested, as
        # is permitted by the interface
        ret = {"positivity": "any", "divisible_by_two": obj % 2 == 0}
        if obj > 0:
            ret["positivity"] = ">0"
        elif obj == 0:
            ret["positivity"] = ">=0"

        return ret


class StrNum(plugin.Wrapper, abstract=MyNumericAbstractType):
    def __init__(self, val):
        self.value = val
        assert isinstance(val, str)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.value == other.value

    @classmethod
    def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
        cls._validate_abstract_props(props)

        value = obj.value
        # only compute properties that were requested
        ret = {}
        for propname in props:
            if propname == "positivity":
                if value.startswith("-"):
                    positivity = "any"
                elif value == "0":
                    positivity = ">=0"
                else:
                    positivity = ">0"
                ret["positivity"] = positivity
            elif propname == "divisible_by_two":
                ret["divisible_by_two"] = int(value) % 2 == 0
        return ret


class StrType(plugin.ConcreteType, abstract=MyAbstractType):
    value_type = str
    allowed_props = dict(lowercase=bool)
    target = "pdp11"

    @classmethod
    def compute_concrete_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
        cls._validate_concrete_props(props)

        # only compute properties that were requested
        ret = {}
        for propname in props:
            if propname == "lowercase":
                ret["lowercase"] = obj.lower() == obj
        return ret


class OtherType(plugin.ConcreteType, abstract=MyAbstractType):
    target = "pdp11"

    @classmethod
    def is_typeclass_of(cls, obj):
        return False  # this type class matches nothing


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
