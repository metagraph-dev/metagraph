import pytest

from metagraph import (
    AbstractType,
    ConcreteType,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)
from metagraph.core.resolver import Resolver, Namespace, Dispatcher

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


def test_dispatcher(example_resolver):
    ns = Dispatcher(example_resolver, "power")
    assert ns(2, 3) == 8


def test_load_plugins(site_dir):
    res = Resolver()
    res.load_plugins_from_environment()

    assert len(res.abstract_types) == 1
    assert len(res.concrete_types) == 2
    assert len(res.translators) == 2
    assert len(res.abstract_algorithms) == 1
    assert len(res.concrete_algorithms) == 1
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

    class Concrete1(ConcreteType):
        abstract = Abstract1

    with pytest.raises(ValueError, match="unregistered abstract"):
        res.register(concrete_types=[Concrete1])

    # forgetting to set abstract attribute is tested in test_plugins now

    class Abstract2(AbstractType):
        pass

    class Concrete2(ConcreteType):
        abstract = Abstract2

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

    @concrete_algorithm("testing.does_not_exist")
    def my_algo3(a: Abstract1) -> Abstract2:  # pragma: no cover
        pass

    with pytest.raises(ValueError, match="unregistered abstract"):
        res.register(concrete_algorithms=[my_algo3])

    class Concrete3(ConcreteType):
        abstract = Abstract1
        value_class = int

    class Concrete4(ConcreteType):
        abstract = Abstract1
        value_class = int

    with pytest.raises(ValueError, match="already has a registered concrete type"):
        res.register(abstract_types=[Abstract1], concrete_types=[Concrete3, Concrete4])


def test_incorrect_signature_errors(example_resolver):
    from .util import IntType

    class Abstract1(AbstractType):
        pass

    class Concrete1(ConcreteType):
        abstract = Abstract1

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


def test_python_types_in_signature(example_resolver):
    from .util import IntType, MyAbstractType

    @abstract_algorithm("testing.python_types")
    def python_types(x: MyAbstractType, p: int) -> MyAbstractType:  # pragma: no cover
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
    def wrong_return_type(x: IntType, p: int) -> float:  # pragma: no cover
        pass

    with pytest.raises(TypeError, match="is not a concrete type of"):
        example_resolver.register(concrete_algorithms=[wrong_return_type])

    @abstract_algorithm("testing.return_type")
    def return_type(x: MyAbstractType) -> int:  # pragma: nocover
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
    from .util import IntType, MyAbstractType

    @abstract_algorithm("testing.python_types")
    def python_types(x: MyAbstractType, p: int) -> MyAbstractType:  # pragma: no cover
        pass

    example_resolver.register(abstract_algorithms=[python_types])

    @concrete_algorithm("testing.python_types")
    def correct_python_type(x: int, p: int) -> IntType:  # pragma: no cover
        pass

    example_resolver.register(concrete_algorithms=[correct_python_type])
    algo = example_resolver.find_algorithm("testing.python_types", 3, 4)
    assert algo == correct_python_type


def test_typeof(example_resolver):
    empty_res = Resolver()
    with pytest.raises(TypeError, match="registered type"):
        t = empty_res.typeof(4)

    from .util import StrType, IntType, OtherType

    assert example_resolver.typeof(4) == IntType()
    assert example_resolver.typeof("python") == StrType(lowercase=True)
    assert example_resolver.typeof("Python") == StrType(lowercase=False)
    assert StrType().is_satisfied_by(example_resolver.typeof("python"))


def test_find_translator(example_resolver):
    from .util import StrType, IntType, OtherType, int_to_str, str_to_int

    assert example_resolver.find_translator(4, StrType) == int_to_str
    assert example_resolver.find_translator("4", IntType) == str_to_int
    assert example_resolver.find_translator(4, OtherType) is None
    assert (
        example_resolver.find_translator(4, IntType) is None
    )  # no self-translator registered


def test_translate(example_resolver):
    from .util import StrType, IntType, OtherType

    assert example_resolver.translate(4, StrType) == "4"
    assert example_resolver.translate("4", IntType) == 4
    with pytest.raises(TypeError, match="Cannot convert"):
        example_resolver.translate(4, OtherType)
    with pytest.raises(TypeError, match="does not have a registered type"):
        example_resolver.translate(set(), StrType)

    # Check registration of translator with Python types works
    @translator
    def int_to_int(src: int, **props) -> int:  # pragma: no cover
        return src

    example_resolver.register(translators=[int_to_int])
    assert example_resolver.translate(4, int) == 4


def test_find_algorithm(example_resolver):
    from .util import int_power, MyAbstractType

    with pytest.raises(ValueError, match='No abstract algorithm "does_not_exist"'):
        example_resolver.find_algorithm("does_not_exist", 1, thing=2)

    assert example_resolver.find_algorithm("power", 1, 3) == int_power
    assert example_resolver.find_algorithm("power", p=1, x=3) == int_power
    assert example_resolver.find_algorithm("power", 1, "4") is None
    assert example_resolver.find_algorithm("power", 1, p=2) == int_power

    with pytest.raises(TypeError, match="too many positional arguments"):
        example_resolver.find_algorithm("power", 1, 2, 3)

    with pytest.raises(TypeError, match="missing a required argument: 'p'"):
        example_resolver.find_algorithm("power", x=1, q=2)

    @abstract_algorithm("testing.match_python_type")
    def python_type(x: MyAbstractType) -> int:  # pragma: no cover
        pass

    @concrete_algorithm("testing.match_python_type")
    def correct_python_type(x: int) -> int:  # pragma: no cover
        return 2 * x

    example_resolver.register(
        abstract_algorithms=[python_type], concrete_algorithms=[correct_python_type]
    )
    algo = example_resolver.find_algorithm("testing.match_python_type", 2)
    assert algo == correct_python_type
    assert example_resolver.find_algorithm("testing.match_python_type", set()) is None


def test_call_algorithm(example_resolver):
    from .util import int_power

    with pytest.raises(ValueError, match='No abstract algorithm "does_not_exist"'):
        example_resolver.call_algorithm("does_not_exist", 1, thing=2)

    assert example_resolver.call_algorithm("power", 2, 3) == 8
    assert example_resolver.call_algorithm("power", p=2, x=3) == 9
    with pytest.raises(
        TypeError,
        match='No concrete algorithm for "power" can be found matching argument type signature',
    ):
        example_resolver.call_algorithm("power", 1, "4")
    assert example_resolver.call_algorithm("power", 2, p=3) == 8


def test_algo_attribute(example_resolver):
    with pytest.raises(
        AttributeError, match="'Namespace' object has no attribute 'does_not_exist'"
    ):
        example_resolver.algo.does_not_exist(1, thing=2)

    assert example_resolver.algo.power(2, 3) == 8
    assert example_resolver.algo.power(p=2, x=3) == 9
    with pytest.raises(
        TypeError,
        match='No concrete algorithm for "power" can be found matching argument type signature',
    ):
        example_resolver.algo.power(1, "4")
