import pytest

from metagraph import (
    AbstractType,
    ConcreteType,
    Wrapper,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)
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
        if isinstance(x, list):
            return any(is_from_plugin1(item) for item in x)
        if hasattr(x, "__wrapped__"):
            x = x.__wrapped__
        return x.__module__.endswith("plugin1")

    assert len([x for x in res.abstract_types if is_from_plugin1(x)]) == 1
    assert len([x for x in res.concrete_types if is_from_plugin1(x)]) == 2
    assert len([x for x in res.translators.values() if is_from_plugin1(x)]) == 2
    assert len([x for x in res.abstract_algorithms.values() if is_from_plugin1(x)]) == 1
    assert len([x for x in res.concrete_algorithms.values() if is_from_plugin1(x)]) == 1
    assert "hyperstuff.supercluster" in res.concrete_algorithms
    assert len(res.concrete_algorithms["hyperstuff.supercluster"]) == 2


def test_register_errors():
    res = Resolver()

    class Abstract1(AbstractType):
        pass

    res.register(abstract_types=[Abstract1])
    with pytest.raises(ValueError, match="already exists"):
        res.register(abstract_types=[Abstract1])

    res = Resolver()

    class Concrete1(ConcreteType, abstract=Abstract1):
        pass

    with pytest.raises(ValueError, match="unregistered abstract"):
        res.register(concrete_types=[Concrete1])

    # forgetting to set abstract attribute is tested in test_plugins now

    class Abstract2(AbstractType):
        pass

    class Concrete2(ConcreteType, abstract=Abstract2):
        pass

    @translator
    def c1_to_c2(src: Concrete1, **props) -> Concrete2:  # pragma: no cover
        pass

    with pytest.raises(ValueError, match="convert between concrete types"):
        res.register(translators=[c1_to_c2])

    @abstract_algorithm("testing.myalgo")
    def my_algo(a: Abstract1) -> Abstract2:  # pragma: no cover
        pass

    @abstract_algorithm("testing.myalgo")
    def my_algo2(a: Abstract1) -> Abstract2:  # pragma: no cover
        pass

    res.register(abstract_algorithms=[my_algo])
    with pytest.raises(ValueError, match="already exists"):
        res.register(abstract_algorithms=[my_algo2])

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
        res.register(abstract_algorithms=[my_algo_bad_input_type])

    with pytest.raises(TypeError, match="return type may not be an instance of"):
        res.register(abstract_algorithms=[my_algo_bad_output_type])

    with pytest.raises(TypeError, match="return type may not be typing.List"):
        res.register(abstract_algorithms=[my_algo_bad_compound_output_type])

    @concrete_algorithm("testing.does_not_exist")
    def my_algo3(a: Abstract1) -> Abstract2:  # pragma: no cover
        pass

    with pytest.raises(ValueError, match="unregistered abstract"):
        res.register(concrete_algorithms=[my_algo3])

    class Concrete3(ConcreteType, abstract=Abstract1):
        value_type = int

    class Concrete4(ConcreteType, abstract=Abstract1):
        value_type = int

    with pytest.raises(ValueError, match="already has a registered concrete type"):
        res.register(abstract_types=[Abstract1], concrete_types=[Concrete3, Concrete4])

    @concrete_algorithm("testing.myalgo")
    def conc_algo_with_defaults(a: Concrete1 = 17) -> Concrete2:  # pragma: no cover
        return a

    with pytest.raises(TypeError, match='argument "a" declares a default value'):
        res.register(concrete_algorithms=[conc_algo_with_defaults])

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
        res.register(
            abstract_algorithms=[my_multi_ret_algo],
            concrete_algorithms=[conc_algo_wrong_output_nums],
        )

    @abstract_algorithm("testing.any")
    def abstract_any(x: Any) -> int:  # pragma: no cover
        pass

    @concrete_algorithm("testing.any")
    def my_any(x: int) -> int:  # pragma: no cover
        return 12

    with pytest.raises(
        TypeError, match='argument "x" does not match abstract function signature'
    ):
        res.register(abstract_algorithms=[abstract_any], concrete_algorithms=[my_any])


def test_incorrect_signature_errors(example_resolver):
    from .util import IntType, FloatType

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
        example_resolver.register(concrete_algorithms=[too_many_args])

    @concrete_algorithm("power")
    def wrong_abstract_arg(x: Concrete1, p: IntType) -> IntType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match='"x" does not have type compatible'):
        example_resolver.register(concrete_algorithms=[wrong_abstract_arg])

    @concrete_algorithm("power")
    def wrong_return_arg(x: IntType, p: IntType) -> Concrete1:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="return type is not compatible"):
        example_resolver.register(concrete_algorithms=[wrong_return_arg])

    @concrete_algorithm("power")
    def wrong_arg_name(X: IntType, p: IntType) -> IntType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match='"X" does not match name of parameter'):
        example_resolver.register(concrete_algorithms=[wrong_arg_name])

    @concrete_algorithm("ln")
    def specifies_abstract_property(
        x: FloatType(positivity=">=0"),
    ) -> FloatType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="specifies abstract properties"):
        example_resolver.register(concrete_algorithms=[specifies_abstract_property])


def test_python_types_in_signature(example_resolver):
    from .util import IntType, MyNumericAbstractType

    @abstract_algorithm("testing.python_types")
    def python_types(
        x: MyNumericAbstractType, p: int
    ) -> MyNumericAbstractType:  # pragma: no cover
        pass

    example_resolver.register(abstract_algorithms=[python_types])

    @concrete_algorithm("testing.python_types")
    def correct_python_type(x: IntType, p: int) -> IntType:  # pragma: no cover
        pass

    example_resolver.register(concrete_algorithms=[correct_python_type])

    @concrete_algorithm("testing.python_types")
    def wrong_python_type(x: IntType, p) -> IntType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match='"p" does not match'):
        example_resolver.register(concrete_algorithms=[wrong_python_type])

    @concrete_algorithm("testing.python_types")
    def wrong_return_type(x: IntType, p: int) -> complex:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="is not a concrete type of"):
        example_resolver.register(concrete_algorithms=[wrong_return_type])

    @abstract_algorithm("testing.return_type")
    def return_type(x: MyNumericAbstractType) -> int:  # pragma: nocover
        pass

    @concrete_algorithm("testing.return_type")
    def notmatching_return_type(x: int) -> float:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="return type does not match"):
        example_resolver.register(
            abstract_algorithms=[return_type],
            concrete_algorithms=[notmatching_return_type],
        )


def test_python_types_as_concrete_substitutes(example_resolver):
    from .util import IntType, MyNumericAbstractType

    @abstract_algorithm("testing.python_types")
    def python_types(
        x: MyNumericAbstractType, p: int
    ) -> MyNumericAbstractType:  # pragma: no cover
        pass

    example_resolver.register(abstract_algorithms=[python_types])

    @concrete_algorithm("testing.python_types")
    def correct_python_type(x: int, p: int) -> IntType:  # pragma: no cover
        pass

    example_resolver.register(concrete_algorithms=[correct_python_type])
    algo_plan = example_resolver.find_algorithm_exact("testing.python_types", 3, 4)
    assert algo_plan.algo == correct_python_type


def test_type_of(example_resolver):
    empty_res = Resolver()
    with pytest.raises(TypeError, match="registered type"):
        t = empty_res.type_of(4)

    from .util import StrType, IntType, OtherType

    assert example_resolver.type_of(4) == IntType()
    assert example_resolver.type_of("python") == StrType(lowercase=True)
    assert example_resolver.type_of("Python") == StrType(lowercase=False)
    assert StrType().is_satisfied_by(example_resolver.type_of("python"))


def test_find_translator(example_resolver):
    from .util import StrNum, IntType, OtherType, int_to_str, str_to_int

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
    from .util import StrNum, IntType, OtherType

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

    example_resolver.register(translators=[int_to_int])
    assert example_resolver.translate(4, int) == 4


def test_translate_plan(example_resolver, capsys):
    from .util import StrNum, OtherType

    capsys.readouterr()
    example_resolver.plan.translate(4, StrNum.Type)
    captured = capsys.readouterr()
    assert captured.out == "[Direct Translation]\nIntType -> StrNumType\n"
    example_resolver.plan.translate(4, OtherType)
    captured = capsys.readouterr()
    assert captured.out == "No translation path found for IntType -> OtherType\n"


def test_find_algorithm(example_resolver):
    from .util import int_power, MyNumericAbstractType

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

    example_resolver.register(
        abstract_algorithms=[python_type], concrete_algorithms=[correct_python_type]
    )
    plan = example_resolver.find_algorithm("testing.match_python_type", 2)
    assert plan.algo == correct_python_type
    assert example_resolver.find_algorithm("testing.match_python_type", set()) is None


def test_call_algorithm(example_resolver):
    from .util import StrNum

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
    assert example_resolver.algo.odict_rev(od1) == od2

    with pytest.raises(TypeError, match="x must be of type"):
        example_resolver.algo.odict_rev(14)


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
    from .util import StrNum

    with config.set({"core.logging.plans": True}):
        assert example_resolver.call_algorithm("power", 2, 3) == 8
    captured = capsys.readouterr()
    assert "int_power" in captured.out

    with config.set({"core.logging.translations": True}):
        assert example_resolver.call_algorithm("power", 2, StrNum("3")) == 8
    captured = capsys.readouterr()
    assert "StrNumType -> IntType" in captured.out


def test_disable_automatic_translation(example_resolver, capsys):
    from .util import StrNum

    with config.set({"core.dispatch.allow_translation": False}):
        with pytest.raises(TypeError) as e:
            example_resolver.call_algorithm("power", 2, StrNum("3"))


def test_algo_attribute(example_resolver):
    with pytest.raises(
        AttributeError, match="'Namespace' object has no attribute 'does_not_exist'"
    ):
        example_resolver.algo.does_not_exist(1, thing=2)

    assert example_resolver.algo.power(2, 3) == 8
    assert example_resolver.algo.power(p=2, x=3) == 9
    with pytest.raises(
        TypeError,
        match="p must be of type MyNumericAbstractType, not MyAbstractType::StrType",
    ):
        example_resolver.algo.power(1, "4")


def test_concrete_algorithm_with_properties(example_resolver):
    from .util import StrNum

    val = example_resolver.algo.ln(100.0)
    assert abs(val - 4.605170185988092) < 1e-6

    with pytest.raises(ValueError, match="does not meet the specificity requirement"):
        example_resolver.algo.ln(-1.1)

    with pytest.raises(ValueError, match="does not meet the specificity requirement"):
        example_resolver.algo.ln(0.0)

    with pytest.raises(ValueError, match="does not meet the specificity requirement"):
        example_resolver.algo.ln(StrNum("0"))


def test_concrete_algorithm_insufficient_specificity(example_resolver):
    from .util import MyNumericAbstractType, FloatType

    class RandomFloatType(Wrapper, abstract=MyNumericAbstractType):
        abstract_property_specificity_limits = {"positivity": "any"}

        def __init__(self):  # pragma: no cover
            import random

            self.value = (random.random() - 0.5) * 100

    example_resolver.register(wrappers=[RandomFloatType])

    @concrete_algorithm("ln")
    def insufficient_ln_function(x: RandomFloatType) -> FloatType:  # pragma: no cover
        import math

        return math.log(x.value)

    # RandomFloatType cannot be restricted to positivity=">0", while the
    # abstract algorithm definition requires such specificity
    with pytest.raises(
        TypeError, match='"positivity" has specificity limits which are incompatible'
    ):
        example_resolver.register(concrete_algorithms=[insufficient_ln_function])
