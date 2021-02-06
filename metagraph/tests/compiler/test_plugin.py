import pytest
from metagraph import Compiler, abstract_algorithm, concrete_algorithm, PluginRegistry
from metagraph.core.plugin import CompileError
from metagraph.tests.util import example_resolver, FailCompiler


def test_top_level_import():
    from metagraph import Compiler


def test_implementation_errors():
    class IncompleteCompiler(Compiler):
        pass

    c = IncompleteCompiler(name="test1")
    c.initialize_runtime()
    c.teardown_runtime()

    with pytest.raises(NotImplementedError) as e:
        c.compile_algorithm(algo=None)

    with pytest.raises(NotImplementedError) as e:
        c.compile_subgraph(subgraph={}, inputs=[], output=None)


def test_compiler_registry(example_resolver):
    res = example_resolver
    assert res.compilers["fail"] is not None

    compiler = res.compilers["fail"]
    assert compiler.name == "fail"

    with pytest.raises(ValueError, match="named 'fail'") as e:
        res.register({"redundant_plugin": {"compilers": {FailCompiler()}}})


def test_register_algorithm(example_resolver):
    res = example_resolver

    @abstract_algorithm("testing.add_two")
    def add_two(x: int) -> int:  # pragma: no cover
        pass

    @concrete_algorithm("testing.add_two", compiler="fail")
    def add_two_c(x: int) -> int:  # pragma: no cover
        return x + 2

    with pytest.raises(CompileError, match="Cannot call"):
        add_two_c(4)

    registry = PluginRegistry("test_register_algorithm")
    registry.register(add_two)
    registry.register(add_two_c)

    example_resolver.register(registry.plugins)
