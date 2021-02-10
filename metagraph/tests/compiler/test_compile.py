import pytest
from metagraph import Compiler, abstract_algorithm, concrete_algorithm, PluginRegistry
from metagraph.core.plugin import CompileError
from metagraph.tests.util import example_resolver, FailCompiler


def test_compile_immediate(example_resolver):
    res = example_resolver

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
