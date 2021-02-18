import pytest
from metagraph import Compiler, abstract_algorithm, concrete_algorithm, PluginRegistry
from metagraph.core.plugin import CompileError
from metagraph.tests.util import example_resolver, FailCompiler


def test_compile_immediate(example_resolver):
    @abstract_algorithm("testing.add_two")
    def add_two(x: int) -> int:  # pragma: no cover
        pass

    @concrete_algorithm("testing.add_two", compiler="identity_comp")
    def add_two_c(x: int) -> int:  # pragma: no cover
        return x + 2

    registry = PluginRegistry("test_register_algorithm")
    registry.register(add_two)
    registry.register(add_two_c)

    example_resolver.register(registry.plugins)

    with pytest.raises(CompileError, match="Cannot call add_two_c"):
        add_two_c(4)

    assert add_two_c(4, resolver=example_resolver) == 6
    assert add_two_c(4) == 6


def test_compile_errors(example_resolver):
    @abstract_algorithm("testing.add_two")
    def add_two(x: int) -> int:  # pragma: no cover
        pass

    @concrete_algorithm("testing.add_two", compiler="foobar")
    def add_two_c(x: int) -> int:  # pragma: no cover
        return x + 2

    @concrete_algorithm("testing.add_two")
    def add_two_noc(x: int) -> int:
        return x + 2

    registry = PluginRegistry("test_register_algorithm")
    registry.register(add_two)
    registry.register(add_two_c)
    registry.register(add_two_noc, name="test_register_algorithm_noc")

    example_resolver.register(registry.plugins)

    with pytest.raises(CompileError, match="Required compiler 'foobar' not found"):
        add_two_c(4, resolver=example_resolver)

    with pytest.raises(
        CompileError, match="Concrete algorithm 'add_two_noc' is not compilable"
    ):
        example_resolver.compile_algorithm(add_two_noc)
