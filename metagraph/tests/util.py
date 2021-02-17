import os
import sys
import pytest
import math
import codecs
from typing import List, Set, Dict, Tuple, Any, Callable, Optional, Union
from collections import OrderedDict

import metagraph as mg
from metagraph import ConcreteType
from metagraph.core import plugin
from metagraph.core.resolver import Resolver
import dask.core


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
    properties = {
        "positivity": ["any", "==0", ">0"],
        "divisible_by_two": [False, True],
        "fizzbuzz": [None, "fizz", "buzz", "fizzbuzz"],
    }


class IntType(plugin.ConcreteType, abstract=MyNumericAbstractType):
    value_type = int
    target = "pdp11"

    @classmethod
    def _compute_abstract_properties(
        cls, obj, props: Set[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        # return all properties regardless of what was requested, as
        # is permitted by the interface
        ret = {}
        ret["positivity"] = ">0" if obj > 0 else "==0" if obj == 0 else "any"
        ret["divisible_by_two"] = obj % 2 == 0
        if obj % 3 == 0 and obj % 5 == 0:
            ret["fizzbuzz"] = "fizzbuzz"
        elif obj % 3 == 0:
            ret["fizzbuzz"] = "fizz"
        elif obj % 5 == 0:
            ret["fizzbuzz"] = "buzz"
        else:
            ret["fizzbuzz"] = None
        return ret

    @classmethod
    def assert_equal(
        cls,
        obj1,
        obj2,
        aprops1,
        aprops2,
        cprops1,
        cprops2,
        *,
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        return obj1 == obj2


class FloatType(plugin.ConcreteType, abstract=MyNumericAbstractType):
    value_type = float
    target = "pdp11"

    @classmethod
    def _compute_abstract_properties(
        cls, obj, props: Set[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        # return all properties regardless of what was requested, as
        # is permitted by the interface
        ret = {}
        ret["positivity"] = ">0" if obj > 0 else "==0" if obj == 0 else "any"
        ret["divisible_by_two"] = obj % 2 == 0
        if obj % 3 == 0 and obj % 5 == 0:
            ret["fizzbuzz"] = "fizzbuzz"
        elif obj % 3 == 0:
            ret["fizzbuzz"] = "fizz"
        elif obj % 5 == 0:
            ret["fizzbuzz"] = "buzz"
        else:
            ret["fizzbuzz"] = None
        return ret

    @classmethod
    def assert_equal(
        cls,
        obj1,
        obj2,
        aprops1,
        aprops2,
        cprops1,
        cprops2,
        *,
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        return math.isclose(obj1, obj2, rel_tol=rel_tol, abs_tol=abs_tol)


class StrNum(plugin.Wrapper, abstract=MyNumericAbstractType):
    def __init__(self, val):
        super().__init__()
        self.value = val
        assert isinstance(val, str)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented  # pragma: no cover
        return self.value == other.value

    def to_num(self):
        try:
            return int(self.value)
        except ValueError:
            return float(self.value)

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
                        positivity = "==0"
                    else:
                        positivity = ">0"
                    ret["positivity"] = positivity
                elif propname == "divisible_by_two":
                    ret["divisible_by_two"] = int(value) % 2 == 0
                elif propname == "fizzbuzz":
                    v = int(value)
                    if v % 3 == 0 and v % 5 == 0:
                        fb = "fizzbuzz"
                    elif v % 3 == 0:
                        fb = "fizz"
                    elif v % 5 == 0:
                        fb = "buzz"
                    else:
                        fb = None
                    ret["fizzbuzz"] = fb
            return ret

        @classmethod
        def assert_equal(
            cls,
            obj1,
            obj2,
            aprops1,
            aprops2,
            cprops1,
            cprops2,
            *,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            return math.isclose(
                float(obj1.value), float(obj2.value), rel_tol=rel_tol, abs_tol=abs_tol
            )


class StrNumRot13(plugin.Wrapper, abstract=MyNumericAbstractType):
    def __init__(self, val):
        super().__init__()
        self.value = val
        assert isinstance(val, str)

    def to_num(self):
        val = codecs.encode(self.value, "rot-13")
        try:
            return int(val)
        except ValueError:
            return float(val)

    class TypeMixin:
        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: Set[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            unrot = codecs.encode(obj.value, "rot-13")
            return FloatType._compute_abstract_properties(unrot, props, known_props)

        @classmethod
        def assert_equal(
            cls,
            obj1,
            obj2,
            aprops1,
            aprops2,
            cprops1,
            cprops2,
            *,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            val1 = codecs.encode(obj1.value, "rot-13")
            val2 = codecs.encode(obj2.value, "rot-13")
            return math.isclose(
                float(val1), float(val2), rel_tol=rel_tol, abs_tol=abs_tol
            )


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


@plugin.translator(include_resolver=True)
def str_to_rot13(src: StrNum, *, resolver) -> StrNumRot13:
    assert resolver is not None
    rot = codecs.encode(src.value, "rot-13")
    return StrNumRot13(rot)


@plugin.abstract_algorithm("power")
def abstract_power(
    x: MyNumericAbstractType, p: MyNumericAbstractType
) -> MyNumericAbstractType:  # pragma: no cover
    """Raise x to the power of p"""
    pass


@plugin.concrete_algorithm("power")
def int_power(x: IntType, p: IntType) -> IntType:
    return x ** p


@plugin.concrete_algorithm("power")
def strnum_power(x: StrNum, p: StrNum) -> StrNum:
    result = x.to_num() ** p.to_num()
    return StrNum(str(result))


@plugin.concrete_algorithm("power", include_resolver=True)
def strnumrot13_power(x: StrNumRot13, p: StrNumRot13, *, resolver) -> StrNumRot13:
    assert resolver is not None
    result = x.to_num() ** p.to_num()
    return StrNumRot13(codecs.encode(str(result), "rot-13"))


@plugin.abstract_algorithm("ln")
def abstract_ln(
    x: MyNumericAbstractType(positivity=">0"),
) -> MyNumericAbstractType:  # pragma: no cover
    """Take the natural log"""
    pass


@plugin.concrete_algorithm("ln")
def float_ln(x: FloatType) -> FloatType:
    return math.log(x)


@plugin.abstract_algorithm("fizzbuzz_club")
def fizzbuzz_club(
    x: MyNumericAbstractType(fizzbuzz=("fizz", "buzz", "fizzbuzz"))
) -> str:
    pass


@plugin.concrete_algorithm("fizzbuzz_club")
def strnum_fizzbuzz_club(x: StrNum) -> str:
    aprops = StrNum.Type.compute_abstract_properties(x, {"fizzbuzz"})
    return f'Hi, {aprops["fizzbuzz"]}'


@plugin.abstract_algorithm("repeat")
def abstract_repeat(
    x: Union[MyAbstractType, MyNumericAbstractType],
    times: Optional[int] = None,
    sep: Optional[str] = None,
) -> str:
    pass


@plugin.concrete_algorithm("repeat")
def string_repeat(
    x: Union[StrType, StrNum], times: Optional[int], sep: Optional[str]
) -> str:
    if isinstance(x, StrNum):
        x = x.value
    if times is None:
        times = 2
    if sep is None:
        sep = ""
    return sep.join([x] * times)


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


@plugin.abstract_algorithm("add_me_up")
def add_me_up(x: List[MyNumericAbstractType]) -> MyNumericAbstractType:
    pass


@plugin.concrete_algorithm("add_me_up")
def simple_add_me_up(x: List[IntType]) -> IntType:
    sum = 0
    for item in x:
        sum += item
    return sum


@plugin.abstract_algorithm("crazy_inputs")
def crazy_inputs(
    a1: mg.NodeID,
    a2: MyNumericAbstractType,
    a3: int,
    b1: Union[int, float],
    b2: mg.Union[int, float],
    c1: Optional[MyNumericAbstractType],
    c2: mg.Optional[MyNumericAbstractType],
    d1: List[MyNumericAbstractType],
    d2: mg.List[MyNumericAbstractType],
    d3: mg.List[float],
    e: Callable[[int, float], str],
) -> Tuple[int, float]:
    pass


@plugin.concrete_algorithm("crazy_inputs")
def simple_crazy_inputs(
    a1: mg.NodeID,
    a2: int,
    a3: int,
    b1: Union[int, float],
    b2: mg.Union[int, float],
    c1: Optional[int],
    c2: mg.Optional[int],
    d1: List[int],
    d2: mg.List[int],
    d3: mg.List[float],
    e: Callable[[int, float], str],
) -> Tuple[int, float]:
    return (3, 4.5)


class TracingCompiler(plugin.Compiler):
    """This compiler traces every call.  Use as subclass for test compilers.

    Attributes:

      initialize_runtime_calls: int - number of initialize_runtime() calls
      teardown_runtime_calls: int - number of teardown_runtime() calls
      compile_algorithm_calls: List[Dict[str, Any]] - List of arguments to compile_algorithm() calls
      compile_subgraph_calls: List[Dict[str, Any]] - List of arguments to compile_subgraph() calls
    """

    def __init__(self, name):
        self.clear_trace()
        super().__init__(name=name)

    def initialize_runtime(self):
        self.initialize_runtime_calls += 1

    def teardown_runtime(self):
        self.teardown_runtime_calls += 1

    def compile_algorithm(self, *args, **kwargs):
        self.compile_algorithm_calls.append((args, kwargs))

    def compile_subgraph(self, *args, **kwargs):
        self.compile_subgraph_calls.append((args, kwargs))

    def clear_trace(self):
        """Clear trace records.  Call at start of test."""
        self.initialize_runtime_calls = 0
        self.teardown_runtime_calls = 0
        self.compile_algorithm_calls = []
        self.compile_subgraph_calls = []


class FailCompiler(TracingCompiler):
    """This compiler always fails to compile every subgraph and function.
    """

    def __init__(self, name="fail"):
        super().__init__(name=name)

    def compile_algorithm(self, *args, **kwargs):
        super().compile_algorithm(**kwargs)
        raise plugin.CompileError("'fail' compiler always fails")

    def compile_subgraph(self, *args, **kwargs):
        super().compile_subgraph(**kwargs)
        raise plugin.CompileError("'fail' compiler always fails")


class IdentityCompiler(TracingCompiler):
    """This compiler returns functions unchanged.
    """

    def __init__(self, name="identity_comp"):
        super().__init__(name=name)

    def compile_algorithm(self, algo, literals):
        super().compile_algorithm(algo, literals)
        return algo.func

    def compile_subgraph(self, *args, **kwargs):
        super().compile_subgraph(*args, **kwargs)

        def compile_inner(subgraph: Dict, inputs: List[str], output: str) -> Callable:
            def apply(func, args, kwargs):
                return func(*args, **kwargs)

            tasks = {}
            for key, task in subgraph.items():
                algo_wrapper, args, kwargs = task
                tasks[key] = (
                    apply,
                    self.compile_algorithm(algo_wrapper.algo, {}),
                    args,
                    kwargs,
                )

            def fused(*args):
                cache = dict(zip(inputs, args))
                return dask.core.get(tasks, output, cache=cache)

            return fused

        return compile_inner(*args, **kwargs)


# Handy for manual testing
def make_example_resolver():
    res = Resolver()
    import metagraph

    res.register(
        {
            "example_plugin": {
                "abstract_types": {MyAbstractType, MyNumericAbstractType},
                "concrete_types": {StrType, IntType, FloatType, OtherType},
                "wrappers": {StrNum, StrNumRot13},
                "translators": {int_to_str, str_to_int, str_to_rot13},
                "abstract_algorithms": {
                    abstract_power,
                    abstract_ln,
                    abstract_echo,
                    odict_reverse,
                    abstract_repeat,
                    fizzbuzz_club,
                    add_me_up,
                    crazy_inputs,
                },
                "concrete_algorithms": {
                    int_power,
                    float_ln,
                    simple_echo,
                    simple_odict_rev,
                    string_repeat,
                    strnum_fizzbuzz_club,
                    simple_add_me_up,
                    simple_crazy_inputs,
                },
                "compilers": {FailCompiler(), IdentityCompiler()},
            },
            "example2_plugin": {"concrete_algorithms": {strnum_power}},
            "example3_plugin": {"concrete_algorithms": {strnumrot13_power}},
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
