import pytest
from metagraph.core.plugin import Compiler
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
        c.compile(subgraph={}, inputs=[])


def test_compiler_registry(example_resolver):
    res = example_resolver
    assert res.compilers["fail"] is not None

    compiler = res.compilers["fail"]
    assert compiler.name == "fail"

    with pytest.raises(ValueError, match="named 'fail'") as e:
        res.register({"redundant_plugin": {"compilers": {FailCompiler()}}})
