import pytest

import metagraph as mg
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
        return x.__module__.endswith("plugin1")

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

    registry = PluginRegistry("test_register_errors_default_plugin")
    registry.register(Concrete1)
    with pytest.raises(ValueError, match="unregistered abstract"):
        res.register(registry.plugins)

    # forgetting to set abstract attribute is tested in test_plugins now

    class Abstract2(AbstractType):
        pass

    class Concrete2(ConcreteType, abstract=Abstract2):
        pass

    @translator
    def c1_to_c2(src: Concrete1, **props) -> Concrete2:  # pragma: no cover
        pass

    registry.register(c1_to_c2)
    with pytest.raises(ValueError, match="has unregistered abstract type "):
        res.register(registry.plugins)

    registry.register(Abstract1)
    with pytest.raises(
        ValueError, match="translator destination type .* has not been registered"
    ):
        res.register(registry.plugins)

    # Fresh start -- too much baggage of things partially registered above
    res = Resolver()

    registry.register(Abstract2)
    registry.register(Concrete2)
    with pytest.raises(ValueError, match="convert between concrete types"):
        res.register(registry.plugins)

    @abstract_algorithm("testing.myalgo")
    def my_algo(a: Abstract1) -> Abstract2:  # pragma: no cover
        pass

    @abstract_algorithm("testing.myalgo")
    def my_algo2(a: Abstract1) -> Abstract2:  # pragma: no cover
        pass

    my_algo_registry = PluginRegistry("test_register_errors_default_plugin")
    my_algo_registry.register(my_algo)
    res.register(my_algo_registry.plugins)
    with pytest.raises(ValueError, match="already exists"):
        my_algo2_registry = PluginRegistry("test_register_errors_default_plugin")
        my_algo2_registry.register(my_algo2)
        res.register(my_algo2_registry.plugins)

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
        registry = PluginRegistry("test_register_errors_default_plugin")
        registry.register(my_algo_bad_input_type)
        res.register(registry.plugins)

    with pytest.raises(TypeError, match="return type may not be an instance of"):
        registry = PluginRegistry("test_register_errors_default_plugin")
        registry.register(my_algo_bad_output_type)
        res.register(registry.plugins)

    with pytest.raises(TypeError, match="return type may not be typing.List"):
        registry = PluginRegistry("test_register_errors_default_plugin")
        registry.register(my_algo_bad_compound_output_type)
        res.register(registry.plugins)

    @concrete_algorithm("testing.does_not_exist")
    def my_algo3(a: Abstract1) -> Abstract2:  # pragma: no cover
        pass

    my_algo3_registry = PluginRegistry("test_register_errors_default_plugin")
    my_algo3_registry.register(my_algo3)
    with pytest.raises(ValueError, match="unregistered abstract"):
        res.register(my_algo3_registry.plugins)

    class Concrete3(ConcreteType, abstract=Abstract1):
        value_type = int

    class Concrete4(ConcreteType, abstract=Abstract1):
        value_type = int

    registry = PluginRegistry("test_register_errors_default_plugin")
    registry.register(Concrete3)
    registry.register(Concrete4)
    registry.register(Abstract1)
    with pytest.raises(ValueError, match=r"abstract type .+ already exists"):
        res.register(registry.plugins)

    @concrete_algorithm("testing.myalgo")
    def conc_algo_with_defaults(a: Concrete1 = 17) -> Concrete2:  # pragma: no cover
        return a

    with pytest.raises(TypeError, match='argument "a" declares a default value'):
        registry = PluginRegistry("test_register_errors_default_plugin")
        registry.register(conc_algo_with_defaults)
        res.register(registry.plugins)

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
        registry = PluginRegistry("test_register_errors_default_plugin")
        registry.register(my_multi_ret_algo)
        registry.register(conc_algo_wrong_output_nums)
        res.register(registry.plugins)

    @abstract_algorithm("testing.any")
    def abstract_any(x: Any) -> int:  # pragma: no cover
        pass

    @concrete_algorithm("testing.any")
    def my_any(x: int) -> int:  # pragma: no cover
        return 12

    with pytest.raises(
        TypeError, match='argument "x" does not match abstract function signature'
    ):
        registry = PluginRegistry("test_register_errors_default_plugin")
        registry.register(abstract_any)
        registry.register(my_any, "my_any_plugin")
        res.register(registry.plugins)


def test_incorrect_signature_errors(example_resolver):
    from .util import IntType, FloatType, abstract_power

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
        registry = PluginRegistry("test_incorrect_signature_errors_default_plugin")
        registry.register(too_many_args)
        example_resolver.register(registry.plugins)

    @concrete_algorithm("power")
    def wrong_abstract_arg(x: Concrete1, p: IntType) -> IntType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match='"x" does not have type compatible'):
        registry = PluginRegistry("test_incorrect_signature_errors_default_plugin")
        registry.register(wrong_abstract_arg)
        example_resolver.register(registry.plugins)

    @concrete_algorithm("power")
    def wrong_return_arg(x: IntType, p: IntType) -> Concrete1:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="return type is not compatible"):
        registry = PluginRegistry("test_incorrect_signature_errors_default_plugin")
        registry.register(wrong_return_arg)
        example_resolver.register(registry.plugins)

    @concrete_algorithm("power")
    def wrong_arg_name(X: IntType, p: IntType) -> IntType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match='"X" does not match name of parameter'):
        registry = PluginRegistry("test_incorrect_signature_errors_default_plugin")
        registry.register(wrong_arg_name)
        example_resolver.register(registry.plugins)

    @concrete_algorithm("ln")
    def specifies_abstract_property(
        x: FloatType(positivity=">=0"),
    ) -> FloatType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="specifies abstract properties"):
        registry = PluginRegistry("test_incorrect_signature_errors_default_plugin")
        registry.register(specifies_abstract_property)
        example_resolver.register(registry.plugins)


def test_python_types_in_signature(example_resolver):
    from .util import IntType, MyNumericAbstractType

    @abstract_algorithm("testing.python_types")
    def python_types(
        x: MyNumericAbstractType, p: int
    ) -> MyNumericAbstractType:  # pragma: no cover
        pass

    registry = PluginRegistry("test_python_types_in_signature_default_plugin")
    registry.register(python_types)
    example_resolver.register(registry.plugins)

    @concrete_algorithm("testing.python_types")
    def wrong_python_type(x: IntType, p) -> IntType:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match='"p" does not match'):
        registry = PluginRegistry("test_python_types_in_signature_default_plugin")
        registry.register(wrong_python_type)
        example_resolver.register(registry.plugins)

    @concrete_algorithm("testing.python_types")
    def wrong_return_type(x: IntType, p: int) -> complex:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="is not a concrete type of"):
        registry = PluginRegistry("test_python_types_in_signature_default_plugin")
        registry.register(wrong_return_type)
        example_resolver.register(registry.plugins)

    @abstract_algorithm("testing.return_type")
    def return_type(x: MyNumericAbstractType) -> int:  # pragma: nocover
        pass

    @concrete_algorithm("testing.return_type")
    def notmatching_return_type(x: int) -> float:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="return type does not match"):
        registry = PluginRegistry("test_python_types_in_signature_default_plugin")
        registry.register(return_type)
        registry.register(notmatching_return_type)
        example_resolver.register(registry.plugins)


def test_python_types_as_concrete_substitutes(example_resolver):
    from .util import IntType, MyNumericAbstractType

    @abstract_algorithm("testing.python_types")
    def python_types(
        x: MyNumericAbstractType, p: int
    ) -> MyNumericAbstractType:  # pragma: no cover
        pass

    registry = PluginRegistry(
        "test_python_types_as_concrete_substitutes_default_plugin"
    )
    registry.register(
        python_types, "test_python_types_as_concrete_substitutes_plugin_1"
    )
    example_resolver.register(registry.plugins)

    @concrete_algorithm("testing.python_types")
    def correct_python_type(x: int, p: int) -> IntType:  # pragma: no cover
        pass

    registry = PluginRegistry(
        "test_python_types_as_concrete_substitutes_default_plugin"
    )
    registry.register(
        correct_python_type, "test_python_types_as_concrete_substitutes_plugin_2"
    )
    example_resolver.register(registry.plugins)
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

    registry = PluginRegistry("test_translate_default_plugin")
    registry.register(int_to_int)
    example_resolver.register(registry.plugins)
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

    registry = PluginRegistry("test_find_algorithm_default_plugin")
    registry.register(python_type)
    registry.register(correct_python_type)
    example_resolver.register(registry.plugins)
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
    from .util import StrNum

    val = example_resolver.algos.ln(100.0)
    assert abs(val - 4.605170185988092) < 1e-6

    with pytest.raises(ValueError, match="does not meet the specificity requirement"):
        example_resolver.algos.ln(-1.1)

    with pytest.raises(ValueError, match="does not meet the specificity requirement"):
        example_resolver.algos.ln(0.0)

    with pytest.raises(ValueError, match="does not meet the specificity requirement"):
        example_resolver.algos.ln(StrNum("0"))


def test_concrete_algorithm_insufficient_specificity(example_resolver):
    from .util import MyNumericAbstractType, FloatType

    class RandomFloatType(Wrapper, abstract=MyNumericAbstractType):
        abstract_property_specificity_limits = {"positivity": "any"}

        def __init__(self):  # pragma: no cover
            import random

            self.value = (random.random() - 0.5) * 100

    example_resolver.register(
        {"insufficient_specificity_plugin": {"wrappers": {RandomFloatType}}}
    )

    @concrete_algorithm("ln")
    def insufficient_ln_function(x: RandomFloatType) -> FloatType:  # pragma: no cover
        import math

        return math.log(x.value)

    # RandomFloatType cannot be restricted to positivity=">0", while the
    # abstract algorithm definition requires such specificity
    with pytest.raises(
        TypeError, match='"positivity" has specificity limits which are incompatible'
    ):
        example_resolver.register(
            {
                "insufficient_specificity_plugin": {
                    "concrete_algorithms": {insufficient_ln_function}
                }
            }
        )


def test_plugin_specific_concrete_algorithms():
    r = mg.resolver
    assert hasattr(r, "plugins")
    assert hasattr(r.plugins, "core_networkx")
    assert set(dir(r.plugins.core_networkx)).issubset(set(dir(r)))

    def _assert_trees_subset(resolver_tree, plugin_tree) -> None:
        assert type(resolver_tree) == type(resolver_tree)
        if resolver_tree == plugin_tree:
            pass
        elif isinstance(resolver_tree, mg.core.resolver.Dispatcher):
            assert hasattr(plugin_tree, "__call__")
        elif isinstance(resolver_tree, set):
            assert plugin_tree.issubset(resolver_tree)
        elif isinstance(resolver_tree, dict):
            assert set(plugin_tree.keys()).issubset(set(resolver_tree.keys()))
            for plugin_tree_key in plugin_tree.keys():
                _assert_trees_subset(
                    resolver_tree[plugin_tree_key], plugin_tree[plugin_tree_key]
                )
        elif isinstance(resolver_tree, Namespace):
            resolver_tree_names = dir(resolver_tree)
            plugin_tree_names = dir(plugin_tree)
            assert set(plugin_tree_names).issubset(resolver_tree_names)
            for plugin_tree_name in plugin_tree_names:
                _assert_trees_subset(
                    getattr(resolver_tree, plugin_tree_name),
                    getattr(plugin_tree, plugin_tree_name),
                )
        else:
            raise ValueError(f"Unexpected type {type(resolver_tree)}")
        return

    tree_names = [
        "abstract_algorithms",
        "abstract_types",
        "algos",
        "concrete_algorithms",
        "concrete_types",
        "translators",
        "types",
        "wrappers",
    ]
    for tree_name in tree_names:
        _assert_trees_subset(
            getattr(r, tree_name), getattr(r.plugins.core_networkx, tree_name),
        )

    import networkx as nx

    # Simple graph with 5 triangles
    # 0 - 1    5 - 6
    # | X |    | /
    # 3 - 4 -- 2 - 7
    # TODO: change this to an EdgeSet once triangle count signature is updated
    simple_graph_data = [
        [0, 1, 1],
        [0, 3, 1],
        [0, 4, 1],
        [1, 3, 1],
        [1, 4, 1],
        [2, 4, 1],
        [2, 5, 1],
        [2, 6, 1],
        [3, 4, 1],
        [5, 6, 1],
        [6, 7, 1],
    ]
    simple_graph = nx.Graph()
    simple_graph.add_weighted_edges_from(simple_graph_data)
    graph = r.wrappers.EdgeMap.NetworkXEdgeMap(simple_graph)
    assert r.algos.cluster.triangle_count(graph) == 5
    assert r.plugins.core_networkx.algos.cluster.triangle_count(graph) == 5
    assert r.algos.cluster.triangle_count.core_networkx(graph) == 5


def test_duplciate_plugin():
    class AbstractType1(AbstractType):
        pass

    class AbstractType2(AbstractType):
        pass

    class ConcreteType1(ConcreteType, abstract=AbstractType1):
        value_type = int
        pass

    class ConcreteType2(ConcreteType, abstract=AbstractType2):
        value_type = int
        pass

    @abstract_algorithm("test_duplciate_plugin.test_abstract_algo")
    def abstract_algo1(input_int: int):
        pass

    @concrete_algorithm("test_duplciate_plugin.test_abstract_algo")
    def concrete_algo1(input_int: int):
        pass

    @concrete_algorithm("test_duplciate_plugin.test_abstract_algo")
    def concrete_algo2(input_int: int):
        pass

    res = Resolver()
    with pytest.raises(
        ValueError, match="Multiple concrete algorithms for abstract algorithm"
    ):
        res.register(
            {
                "bad_many_conc_algo_to_one_abstract_algo": {
                    "abstract_algorithms": {abstract_algo1},
                    "concrete_algorithms": {concrete_algo1, concrete_algo2},
                }
            }
        )

    res = Resolver()
    res.register(
        (
            {
                "bad_many_value_type_to_concrete": {
                    "abstract_types": {AbstractType1, AbstractType2},
                    "concrete_types": {ConcreteType1},
                }
            }
        )
    )
    with pytest.raises(
        ValueError,
        match="Python class '<class 'int'>' already has a registered concrete type: ",
    ):
        res.register(
            ({"bad_many_value_type_to_concrete": {"concrete_types": {ConcreteType2}}})
        )

    res = Resolver()
    res.register({"test_duplciate_plugin": {"abstract_types": {AbstractType1}}})
    with pytest.raises(ValueError, match=" already registered."):
        res.register({"test_duplciate_plugin": {"concrete_types": {ConcreteType1}}})

    res = Resolver()
    with pytest.raises(ValueError, match=" not known to be the resolver or a plugin."):
        res._register_plugin_attributes_in_tree(
            Resolver(), abstract_types={AbstractType1}
        )


def test_invalid_plugin_names():
    res = Resolver()
    invalid_plugin_name = "invalid_name!#@$%#^&*()[]"

    class Abstract1(AbstractType):
        pass

    class Concrete1(ConcreteType, abstract=Abstract1):
        pass

    with pytest.raises(ValueError, match="is not a valid plugin name"):
        registry = PluginRegistry(invalid_plugin_name)

    registry = PluginRegistry("test_invalid_plugin_names_default_plugin")
    with pytest.raises(ValueError, match="is not a valid plugin name"):
        registry.register(Abstract1, invalid_plugin_name)

    with pytest.raises(ValueError, match="is not a valid plugin name"):
        registry.register_from_modules(
            mg.types, mg.algorithms, name=invalid_plugin_name
        )

    with pytest.raises(ValueError, match="is not a valid plugin name"):
        res.register({invalid_plugin_name: {"abstract_types": {Abstract1}}})
