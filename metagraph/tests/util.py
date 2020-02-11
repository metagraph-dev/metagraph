import os
import sys

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


class IntType(plugin.ConcreteType):
    abstract = MyAbstractType
    value_class = int
    target = "pdp11"


class StrType(plugin.ConcreteType):
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


class OtherType(plugin.ConcreteType):
    abstract = MyAbstractType
    target = "pdp11"


@plugin.translator
def int_to_str(src: IntType) -> StrType:
    return str(src)


@plugin.translator
def str_to_int(src: StrType) -> IntType:
    return int(src)


@plugin.abstract_algorithm("power")
def abstract_power(
    x: MyAbstractType, p: MyAbstractType
) -> MyAbstractType:  # pragma: no cover
    pass


@plugin.concrete_algorithm("power")
def int_power(x: IntType, p: IntType) -> IntType:
    return x ** p


@pytest.fixture
def example_resolver():
    res = Resolver()
    res.register(
        abstract_types=[MyAbstractType],
        concrete_types=[StrType, IntType, OtherType],
        translators=[int_to_str, str_to_int],
        abstract_algorithms=[abstract_power],
        concrete_algorithms=[int_power],
    )
    return res
