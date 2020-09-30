import os
import sys
import pytest
import math
from typing import List, Set, Dict, Any
from collections import OrderedDict

from metagraph import ConcreteType
from metagraph.core import plugin
from metagraph.core.resolver import Resolver


def make_site_dir_fixture(site_dir):
    test_site_dir = os.path.join(os.path.dirname(__file__), site_dir)
    sys.path.insert(0, test_site_dir)
    yield test_site_dir
    sys.path.remove(test_site_dir)


@pytest.fixture
def site_dir():
    yield from make_site_dir_fixture("site_dir")


@pytest.fixture
def bad_site_dir():
    yield from make_site_dir_fixture("bad_site_dir")


@pytest.fixture
def bad_site_dir2():
    yield from make_site_dir_fixture("bad_site_dir2")


class MyAbstractType(plugin.AbstractType):
    pass


class MyNumericAbstractType(plugin.AbstractType):
    properties = {"positivity": ["any", ">=0", ">0"], "divisible_by_two": [False, True]}


class IntType(plugin.ConcreteType, abstract=MyNumericAbstractType):
    value_type = int
    target = "pdp11"

    @classmethod
    def _compute_abstract_properties(
        cls, obj, props: Set[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
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
    def _compute_abstract_properties(
        cls, obj, props: Set[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            return NotImplemented  # pragma: no cover
        return self.value == other.value

    class TypeMixin:
        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: Set[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:

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
    def _compute_concrete_properties(
        cls, obj, props: List[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:

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


@plugin.abstract_algorithm("echo_str")
def abstract_echo(x: Any, suffix: Any = " <echo>") -> str:  # pragma: no cover
    pass


@plugin.concrete_algorithm("echo_str")
def simple_echo(x: Any, suffix: Any, prefix=None) -> str:  # pragma: no cover
    if prefix:
        return f"{prefix}{x}{suffix}"
    return f"{x}{suffix}"


@plugin.abstract_algorithm("odict_rev")
def odict_reverse(x: OrderedDict) -> OrderedDict:  # pragma: no cover
    pass


@plugin.concrete_algorithm("odict_rev")
def simple_odict_rev(x: OrderedDict) -> OrderedDict:  # pragma: no cover
    d = OrderedDict()
    for k in reversed(x):
        d[k] = x[k]
    return d


# Handy for manual testing
def make_example_resolver():
    res = Resolver()
    import metagraph

    res.register(
        {
            "example_plugin": {
                "abstract_types": {MyAbstractType, MyNumericAbstractType},
                "concrete_types": {StrType, IntType, FloatType, OtherType},
                "wrappers": {StrNum},
                "translators": {int_to_str, str_to_int},
                "abstract_algorithms": {
                    abstract_power,
                    abstract_ln,
                    abstract_echo,
                    odict_reverse,
                },
                "concrete_algorithms": {
                    int_power,
                    float_ln,
                    simple_echo,
                    simple_odict_rev,
                },
            }
        }
    )
    return res


@pytest.fixture
def example_resolver():
    return make_example_resolver()


@pytest.fixture(scope="session")
def default_plugin_resolver(request):  # pragma: no cover
    res = Resolver()
    if request.config.getoption("--no-plugins", default=False):
        from metagraph.plugins import find_plugins

        res.register(**find_plugins())
    else:
        res.load_plugins_from_environment()

    if request.config.getoption("--dask", default=False):
        from metagraph.dask import DaskResolver

        res = DaskResolver(res)

    return res
