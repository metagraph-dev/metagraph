import pytest

from metagraph import (
    AbstractType,
    ConcreteType,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)
from metagraph.core.resolver import Resolver

from .util import site_dir, example_resolver


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


def test_register_abstract_type_error():
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
