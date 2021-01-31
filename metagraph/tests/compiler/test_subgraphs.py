import metagraph.core.compiler as mg_compiler
from dask.delayed import delayed
from metagraph.tests.util import default_plugin_resolver, IdentityCompiler
from metagraph import translator, abstract_algorithm, concrete_algorithm
import networkx as nx
from metagraph.core.resolver import Resolver
from metagraph.core.dask.resolver import DaskResolver
from metagraph import PluginRegistry
from pytest import fixture
import numpy as np


def test_dask_subgraph():
    @delayed
    def func1(x):  # pragma: no cover
        return x + 1

    z = func1(func1(1))

    subgraph = mg_compiler.DaskSubgraph(tasks=z.dask, input_keys=[], output_key=z.key)
    assert len(subgraph.tasks) == 2
    assert len(subgraph.input_keys) == 0
    assert isinstance(subgraph.output_key, str)


def test_extract_subgraphs_noop():
    @delayed
    def func1(x):  # pragma: no cover
        return x + 1

    z = func1(func1(1))

    subgraphs = mg_compiler.extract_compilable_subgraphs(z.dask, compiler="noexist")
    assert len(subgraphs) == 0


def test_extract_subgraphs_chain(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z = scale_func(scale_func(scale_func(a, 2.0), 3.0), 4.0)

    subgraphs = mg_compiler.extract_compilable_subgraphs(
        z.__dask_graph__(), compiler="identity"
    )
    assert len(subgraphs) == 1
    subgraph = subgraphs[0]
    assert len(subgraph.tasks) == 3
    # FIXME: This is zero because the input numpy array is not wrapped in its own placeholder object
    assert len(subgraph.input_keys) == 0
    # FIXME: key property
    assert subgraph.output_key == z.__dask_keys__()[0]


def test_extract_subgraphs_two_chains(res):
    """Two chains feeding into a combining node"""
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z1 = scale_func(scale_func(scale_func(a, 2.0), 3.0), 4.0)
    z2 = scale_func(scale_func(scale_func(a, 2.5), 3.5), 4.5)
    merge = res.algos.testing.add(z1, z2)

    # The merge node cannot be fused with z1 or z2 without reducing parallelism in the graph

    subgraphs = mg_compiler.extract_compilable_subgraphs(
        merge.__dask_graph__(), compiler="identity"
    )
    assert len(subgraphs) == 2
    for subgraph in subgraphs:
        assert len(subgraph.tasks) == 3
        # FIXME: This is zero because the input numpy array is not wrapped in its own placeholder object
        assert len(subgraph.input_keys) == 0
        # we don't know what order the two chains will come out in
        # FIXME: key property
        assert subgraph.output_key in (z1.__dask_keys__()[0], z2.__dask_keys__()[0])


def test_extract_subgraphs_three_chains(res):
    """Two chains feeding into a third chain"""
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z1 = scale_func(scale_func(scale_func(a, 2.0), 3.0), 4.0)
    z2 = scale_func(scale_func(scale_func(a, 2.5), 3.5), 4.5)
    merge = res.algos.testing.add(z1, z2)
    ans = scale_func(merge, 2.8)

    # The merge node cannot be fused with z1 or z2 without reducing parallelism in the graph,
    # but the merge node can start the final chain

    subgraphs = mg_compiler.extract_compilable_subgraphs(
        ans.__dask_graph__(), compiler="identity"
    )
    assert len(subgraphs) == 3

    # separate the final chain from the input chains
    final_chain = None
    input_chains = []
    for subgraph in subgraphs:
        # FIXME: key property
        if subgraph.output_key == ans.__dask_keys__()[0]:
            assert (
                final_chain is None
            ), "found more than one subgraph with key of final chain"
            final_chain = subgraph
        else:
            input_chains.append(subgraph)

    # final chain tests
    assert len(final_chain.tasks) == 2
    assert len(final_chain.input_keys) == 2
    for input_key in final_chain.input_keys:
        # FIXME: key property
        assert input_key in (z1.__dask_keys__()[0], z2.__dask_keys__()[0])
    assert (
        final_chain.output_key == ans.__dask_keys__()[0]
    )  # true by construction, checked here for completeness

    # input_chain tests
    for subgraph in input_chains:
        assert len(subgraph.tasks) == 3
        # FIXME: This is zero because the input numpy array is not wrapped in its own placeholder object
        assert len(subgraph.input_keys) == 0
        # we don't know what order the two chains will come out in
        # FIXME: key property
        assert subgraph.output_key in (z1.__dask_keys__()[0], z2.__dask_keys__()[0])


@fixture
def res():
    from metagraph.plugins.core.types import Vector
    from metagraph.plugins.numpy.types import NumpyVectorType

    @abstract_algorithm("testing.add")
    def testing_add(a: Vector, b: Vector) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.add", compiler="identity")
    def compiled_add(a: NumpyVectorType, b: NumpyVectorType) -> NumpyVectorType:
        return a + b

    @abstract_algorithm("testing.scale")
    def testing_scale(a: Vector, scale: float) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.scale", compiler="identity")
    def compiled_scale(a: NumpyVectorType, scale: float) -> NumpyVectorType:
        return a * scale

    registry = PluginRegistry("test_subgraphs_plugin")
    registry.register(testing_add)
    registry.register(compiled_add)
    registry.register(testing_scale)
    registry.register(compiled_scale)

    registry.register(IdentityCompiler())

    resolver = Resolver()
    resolver.load_plugins_from_environment()
    resolver.register(registry.plugins)

    return DaskResolver(resolver)
