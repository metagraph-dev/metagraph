import pytest

from metagraph import (
    AbstractType,
    ConcreteType,
    Wrapper,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)
from metagraph.core.plugin_registry import PluginRegistry
from metagraph.core.resolver import Resolver, Namespace, Dispatcher, NamespaceError
from metagraph.core.planning import MultiStepTranslator, AlgorithmPlan
from metagraph import config
from typing import Tuple, List, Any
from collections import OrderedDict

from .util import site_dir, example_resolver


def test_namespace():
    ns = Namespace()
    ns._register("A.B.c", 3)
    ns._register("A.B.d", "test")
    ns._register("A.other", 1.5)

    assert ns.A.B.c == 3
    assert ns.A.B.d == "test"
    assert ns.A.other == 1.5
    assert dir(ns) == ["A"]
    assert dir(ns.A) == ["B", "other"]
    assert dir(ns.A.B) == ["c", "d"]

    with pytest.raises(AttributeError, match="does_not_exist"):
        ns.does_not_exist

    with pytest.raises(AttributeError, match="does_not_exist"):
        ns.A.does_not_exist

    with pytest.raises(AttributeError, match="foo"):
        ns.foo.does_not_exist

    with pytest.raises(NamespaceError, match="Name already registered"):
        ns._register("A.B.c", 4)


def test_dispatcher(example_resolver):
    ns = Dispatcher(example_resolver, "power")
    assert ns(2, 3) == 8


def test_load_plugins(site_dir):
    res = Resolver()
    res.load_plugins_from_environment()

    def is_from_plugin1(x):
        if isinstance(x, set):
            return any(is_from_plugin1(item) for item in x)
        if hasattr(x, "__wrapped__"):
            x = x.__wrapped__
        return x.__module__.endswith("plugin1_util")

    assert len([x for x in res.abstract_types if is_from_plugin1(x)]) == 1
    assert len([x for x in res.concrete_types if is_from_plugin1(x)]) == 2
    assert len([x for x in res.translators.values() if is_from_plugin1(x)]) == 2
    assert len([x for x in res.abstract_algorithms.values() if is_from_plugin1(x)]) == 1
    assert len([x for x in res.concrete_algorithms.values() if is_from_plugin1(x)]) == 1
    assert "hyperstuff.supercluster" in res.concrete_algorithms


def test_register_errors():
    res = Resolver()

    class Abstract1(AbstractType):
        pass

    class Concrete1(ConcreteType, abstract=Abstract1):
        pass

    registry = PluginRegistry()
    registry.register_concrete("test_plugin", Concrete1)
    with pytest.raises(ValueError, match="unregistered abstract"):
        res.register(registry)

    # forgetting to set abstract attribute is tested in test_plugins now

    class Abstract2(AbstractType):
        pass

    class Concrete2(ConcreteType, abstract=Abstract2):
        pass

    @translator
    def c1_to_c2(src: Concrete1, **props) -> Concrete2:  # pragma: no cover
        pass

    registry.register_concrete("test_plugin", c1_to_c2)
    with pytest.raises(ValueError, match="has unregistered abstract type "):
        res.register(registry)

    registry.register_abstract(Abstract1)
    with pytest.raises(ValueError, match="convert between concrete types"):
        res.register(registry)

    @abstract_algorithm("testing.myalgo")
    def my_algo(a: Abstract1) -> Abstract2:  # pragma: no cover
        pass

    @abstract_algorithm("testing.myalgo")
    def my_algo2(a: Abstract1) -> Abstract2:  # pragma: no cover
        pass

    my_algo_registry = PluginRegistry()
    my_algo_registry.register_abstract(my_algo)
    res.register(my_algo_registry)
    with pytest.raises(ValueError, match="already exists"):
        my_algo2_registry = PluginRegistry()
        my_algo2_registry.register_abstract(my_algo2)
        res.register(my_algo2_registry)

    @abstract_algorithm("testing.bad_input_type")
    def my_algo_bad_input_type(a: List) -> Resolver:  # pragma: no cover
        pass

    @abstract_algorithm("testing.bad_output_type")
    def my_algo_bad_output_type(a: Abstract1) -> res:  # pragma: no cover
        pass

    @abstract_algorithm("testing.bad_compound_output_type")
    def my_algo_bad_compound_output_type(
        a: Abstract1,
    ) -> Tuple[List, List]:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match='argument "a" may not be typing.List'):
        registry = PluginRegistry()
        registry.register_abstract(my_algo_bad_input_type)
        res.register(registry)

    with pytest.raises(TypeError, match="return type may not be an instance of"):
        registry = PluginRegistry()
        registry.register_abstract(my_algo_bad_output_type)
        res.register(registry)

    with pytest.raises(TypeError, match="return type may not be typing.List"):
        registry = PluginRegistry()
        registry.register_abstract(my_algo_bad_compound_output_type)
        res.register(registry)

    @concrete_algorithm("testing.does_not_exist")
    def my_algo3(a: Abstract1) -> Abstract2:  # pragma: no cover
        pass

    my_algo3_registry = PluginRegistry()
    my_algo3_registry.register_concrete("test_plugin", my_algo3)
    with pytest.raises(ValueError, match="unregistered abstract"):
        res.register(my_algo3_registry)

    class Concrete3(ConcreteType, abstract=Abstract1):
        value_type = int

    class Concrete4(ConcreteType, abstract=Abstract1):
        value_type = int

    registry = PluginRegistry()
    registry.register_concrete("test_plugin", Concrete3)
    registry.register_concrete("test_plugin", Concrete4)
    registry.register_abstract(Abstract1)
    with pytest.raises(ValueError, match=r"abstract type .+ already exists"):
        res.register(registry)

    @concrete_algorithm("testing.myalgo")
    def conc_algo_with_defaults(a: Concrete1 = 17) -> Concrete2:  # pragma: no cover
        return a

    with pytest.raises(TypeError, match='argument "a" declares a default value'):
        registry = PluginRegistry()
        registry.register_concrete(
            "conc_algo_with_defaults_plugin", conc_algo_with_defaults
        )
        res.register(registry)

    @abstract_algorithm("testing.multi_ret")
    def my_multi_ret_algo() -> Tuple[int, int, int]:  # pragma: no cover
        pass

    @concrete_algorithm("testing.multi_ret")
    def conc_algo_wrong_output_nums() -> Tuple[int, int]:  # pragma: no cover
        return (0, 100)

    with pytest.raises(
        TypeError,
        match="return type is not compatible with abstract function signature",
    ):
        registry = PluginRegistry()
        registry.register_abstract(my_multi_ret_algo)
        registry.register_concrete(
            "conc_algo_wrong_output_nums_plugin", conc_algo_wrong_output_nums
        )
        res.register(registry)

    @abstract_algorithm("testing.any")
    def abstract_any(x: Any) -> int:  # pragma: no cover
        pass

    @concrete_algorithm("testing.any")
    def my_any(x: int) -> int:  # pragma: no cover
        return 12

    with pytest.raises(
        TypeError, match='argument "x" does not match abstract function signature'
    ):
        registry = PluginRegistry()
        registry.register_abstract(abstract_any)
        registry.register_concrete("my_any_plugin", my_any)
        res.register(registry)


def test_incorrect_signature_errors(example_resolver):
    from .example_plugin_util import IntType, FloatType, abstract_power

    class Abstract1(AbstractType):
        pass

    class Concrete1(ConcreteType, abstract=Abstract1):
        pass

    @concrete_algorithm("power")
    def too_many_args(
        x: IntType, p: IntType, w: IntType
    ) -> IntType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="number of parameters"):
        registry = PluginRegistry()
        registry.register_concrete("incorrect_signature_errors_plugin", too_many_args)
        example_resolver.register(registry)

    @concrete_algorithm("power")
    def wrong_abstract_arg(x: Concrete1, p: IntType) -> IntType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match='"x" does not have type compatible'):
        registry = PluginRegistry()
        registry.register_concrete(
            "incorrect_signature_errors_plugin", wrong_abstract_arg
        )
        example_resolver.register(registry)

    @concrete_algorithm("power")
    def wrong_return_arg(x: IntType, p: IntType) -> Concrete1:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="return type is not compatible"):
        registry = PluginRegistry()
        registry.register_concrete(
            "incorrect_signature_errors_plugin", wrong_return_arg
        )
        example_resolver.register(registry)

    @concrete_algorithm("power")
    def wrong_arg_name(X: IntType, p: IntType) -> IntType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match='"X" does not match name of parameter'):
        registry = PluginRegistry()
        registry.register_concrete("incorrect_signature_errors_plugin", wrong_arg_name)
        example_resolver.register(registry)

    @concrete_algorithm("ln")
    def specifies_abstract_property(
        x: FloatType(positivity=">=0"),
    ) -> FloatType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="specifies abstract properties"):
        registry = PluginRegistry()
        registry.register_concrete(
            "incorrect_signature_errors_plugin", specifies_abstract_property
        )
        example_resolver.register(registry)


def test_python_types_in_signature(example_resolver):
    from .example_plugin_util import IntType, MyNumericAbstractType

    @abstract_algorithm("testing.python_types")
    def python_types(
        x: MyNumericAbstractType, p: int
    ) -> MyNumericAbstractType:  # pragma: no cover
        pass

    registry = PluginRegistry()
    registry.register_abstract(python_types)
    example_resolver.register(registry)

    @concrete_algorithm("testing.python_types")
    def wrong_python_type(x: IntType, p) -> IntType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match='"p" does not match'):
        registry = PluginRegistry()
        registry.register_concrete(
            "python_types_in_signature_plugin", wrong_python_type
        )
        example_resolver.register(registry)

    @concrete_algorithm("testing.python_types")
    def wrong_return_type(x: IntType, p: int) -> complex:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="is not a concrete type of"):
        registry = PluginRegistry()
        registry.register_concrete(
            "python_types_in_signature_plugin", wrong_return_type
        )
        example_resolver.register(registry)

    @abstract_algorithm("testing.return_type")
    def return_type(x: MyNumericAbstractType) -> int:  # pragma: nocover
        pass

    @concrete_algorithm("testing.return_type")
    def notmatching_return_type(x: int) -> float:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="return type does not match"):
        registry = PluginRegistry()
        registry.register_abstract(return_type)
        registry.register_concrete(
            "python_types_in_signature_plugin", notmatching_return_type
        )
        example_resolver.register(registry)


def test_python_types_as_concrete_substitutes(example_resolver):
    from .example_plugin_util import IntType, MyNumericAbstractType

    @abstract_algorithm("testing.python_types")
    def python_types(
        x: MyNumericAbstractType, p: int
    ) -> MyNumericAbstractType:  # pragma: no cover
        pass

    registry = PluginRegistry()
    registry.register_abstract(python_types)
    example_resolver.register(registry)

    @concrete_algorithm("testing.python_types")
    def correct_python_type(x: int, p: int) -> IntType:  # pragma: no cover
        pass

    registry = PluginRegistry()
    registry.register_concrete("python_types_in_signature_plugin", correct_python_type)
    example_resolver.register(registry)
    algo_plan = example_resolver.find_algorithm_exact("testing.python_types", 3, 4)
    assert algo_plan.algo == correct_python_type


def test_type_of(example_resolver):
    empty_res = Resolver()
    with pytest.raises(TypeError, match="registered type"):
        t = empty_res.type_of(4)

    from .example_plugin_util import StrType, IntType, OtherType

    assert example_resolver.type_of(4) == IntType()
    assert example_resolver.type_of("python") == StrType(lowercase=True)
    assert example_resolver.type_of("Python") == StrType(lowercase=False)
    assert StrType().is_satisfied_by(example_resolver.type_of("python"))


def test_find_translator(example_resolver):
    from .example_plugin_util import StrNum, IntType, OtherType, int_to_str, str_to_int

    def find_translator(value, dst_type):
        src_type = example_resolver.typeclass_of(value)
        trns = MultiStepTranslator.find_translation(
            example_resolver, src_type, dst_type, exact=True
        )
        if trns is not None:
            assert len(trns.translators) == 1
            return trns.translators[0]

    assert find_translator(4, StrNum.Type) == int_to_str
    assert find_translator(StrNum("4"), IntType) == str_to_int
    assert find_translator(4, OtherType) is None
    assert find_translator(4, IntType) is None  # no self-translator registered


def test_translate(example_resolver):
    from .example_plugin_util import StrNum, IntType, OtherType

    assert example_resolver.translate(4, StrNum.Type) == StrNum("4")
    assert example_resolver.translate(StrNum("4"), IntType) == 4
    with pytest.raises(TypeError, match="Cannot convert"):
        example_resolver.translate(4, OtherType)
    with pytest.raises(TypeError, match="does not have a registered type"):
        example_resolver.translate(set(), StrNum.Type)

    # Check registration of translator with Python types works
    @translator
    def int_to_int(src: int, **props) -> int:  # pragma: no cover
        return src

    registry = PluginRegistry()
    registry.register_concrete("translate_plugin", int_to_int)
    example_resolver.register(registry)
    assert example_resolver.translate(4, int) == 4


def test_translate_plan(example_resolver, capsys):
    from .example_plugin_util import StrNum, OtherType

    capsys.readouterr()
    example_resolver.plan.translate(4, StrNum.Type)
    captured = capsys.readouterr()
    assert captured.out == "[Direct Translation]\nIntType -> StrNumType\n"
    example_resolver.plan.translate(4, OtherType)
    captured = capsys.readouterr()
    assert captured.out == "No translation path found for IntType -> OtherType\n"


def test_find_algorithm(example_resolver):
    from .example_plugin_util import int_power, MyNumericAbstractType

    with pytest.raises(ValueError, match='No abstract algorithm "does_not_exist"'):
        example_resolver.find_algorithm("does_not_exist", 1, thing=2)

    assert example_resolver.find_algorithm("power", 1, 3).algo == int_power
    assert example_resolver.find_algorithm("power", p=1, x=3).algo == int_power
    assert example_resolver.find_algorithm("power", 1, "4") is None
    assert example_resolver.find_algorithm("power", 1, p=2).algo == int_power

    with pytest.raises(TypeError, match="too many positional arguments"):
        example_resolver.find_algorithm("power", 1, 2, 3)

    with pytest.raises(TypeError, match="missing a required argument: 'p'"):
        example_resolver.find_algorithm("power", x=1, q=2)

    @abstract_algorithm("testing.match_python_type")
    def python_type(x: MyNumericAbstractType) -> int:  # pragma: no cover
        pass

    @concrete_algorithm("testing.match_python_type")
    def correct_python_type(x: int) -> int:  # pragma: no cover
        return 2 * x

    registry = PluginRegistry()
    registry.register_abstract(python_type)
    registry.register_concrete("translate_plugin", correct_python_type)
    example_resolver.register(registry)
    plan = example_resolver.find_algorithm("testing.match_python_type", 2)
    assert plan.algo == correct_python_type
    assert example_resolver.find_algorithm("testing.match_python_type", set()) is None


def test_call_algorithm(example_resolver):
    from .example_plugin_util import StrNum

    with pytest.raises(ValueError, match='No abstract algorithm "does_not_exist"'):
        example_resolver.call_algorithm("does_not_exist", 1, thing=2)

    assert example_resolver.call_algorithm("power", 2, 3) == 8
    assert example_resolver.call_algorithm("power", p=2, x=3) == 9
    with pytest.raises(
        TypeError,
        match="p must be of type MyNumericAbstractType, not MyAbstractType::StrType",
    ):
        example_resolver.call_algorithm("power", 1, "4")
    assert example_resolver.call_algorithm("power", 2, p=3) == 8
    assert example_resolver.call_algorithm("power", 2, StrNum("3")) == 8
    assert example_resolver.call_algorithm("echo", 14) == 14

    od1 = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
    od2 = OrderedDict([("c", 3), ("b", 2), ("a", 1)])
    assert example_resolver.algos.odict_rev(od1) == od2

    with pytest.raises(TypeError, match="x must be of type"):
        example_resolver.algos.odict_rev(14)


def test_call_algorithm_plan(example_resolver, capsys):
    capsys.readouterr()
    example_resolver.plan.call_algorithm("power", 2, 3)
    captured = capsys.readouterr()
    assert "int_power" in captured.out
    assert "Argument Translations" in captured.out
    example_resolver.plan.call_algorithm("power", 2, "4")
    captured = capsys.readouterr()
    assert (
        'No concrete algorithm for "power" can be satisfied for the given inputs'
        in captured.out
    )


def test_call_algorithm_logging(example_resolver, capsys):
    from .example_plugin_util import StrNum

    with config.set({"core.logging.plans": True}):
        assert example_resolver.call_algorithm("power", 2, 3) == 8
    captured = capsys.readouterr()
    assert "int_power" in captured.out

    with config.set({"core.logging.translations": True}):
        assert example_resolver.call_algorithm("power", 2, StrNum("3")) == 8
    captured = capsys.readouterr()
    assert "StrNumType -> IntType" in captured.out


def test_disable_automatic_translation(example_resolver, capsys):
    from .example_plugin_util import StrNum

    with config.set({"core.dispatch.allow_translation": False}):
        with pytest.raises(TypeError) as e:
            example_resolver.call_algorithm("power", 2, StrNum("3"))


def test_algos_attribute(example_resolver):
    with pytest.raises(
        AttributeError, match="'Namespace' object has no attribute 'does_not_exist'"
    ):
        example_resolver.algos.does_not_exist(1, thing=2)

    assert example_resolver.algos.power(2, 3) == 8
    assert example_resolver.algos.power(p=2, x=3) == 9
    with pytest.raises(
        TypeError,
        match="p must be of type MyNumericAbstractType, not MyAbstractType::StrType",
    ):
        example_resolver.algos.power(1, "4")


def test_concrete_algorithm_with_properties(example_resolver):
    from .example_plugin_util import StrNum

    val = example_resolver.algos.ln(100.0)
    assert abs(val - 4.605170185988092) < 1e-6

    with pytest.raises(ValueError, match="does not meet the specificity requirement"):
        example_resolver.algos.ln(-1.1)

    with pytest.raises(ValueError, match="does not meet the specificity requirement"):
        example_resolver.algos.ln(0.0)

    with pytest.raises(ValueError, match="does not meet the specificity requirement"):
        example_resolver.algos.ln(StrNum("0"))


def test_concrete_algorithm_insufficient_specificity(example_resolver):
    from .example_plugin_util import MyNumericAbstractType, FloatType

    class RandomFloatType(Wrapper, abstract=MyNumericAbstractType):
        abstract_property_specificity_limits = {"positivity": "any"}

        def __init__(self):  # pragma: no cover
            import random

            self.value = (random.random() - 0.5) * 100

    registry = PluginRegistry()
    registry.register_abstract(MyNumericAbstractType)
    registry.register_concrete(
        "concrete_algorithm_insufficient_specificity_plugin", RandomFloatType
    )
    example_resolver.register(registry)

    @concrete_algorithm("ln")
    def insufficient_ln_function(x: RandomFloatType) -> FloatType:  # pragma: no cover
        import math

        return math.log(x.value)

    registry = PluginRegistry()
    registry.register_concrete(
        "concrete_algorithm_insufficient_specificity_plugin", insufficient_ln_function
    )
    # RandomFloatType cannot be restricted to positivity=">0", while the
    # abstract algorithm definition requires such specificity
    with pytest.raises(
        TypeError, match='"positivity" has specificity limits which are incompatible'
    ):
        example_resolver.register(registry)
