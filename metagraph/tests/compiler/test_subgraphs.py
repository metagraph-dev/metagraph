import metagraph.core.compiler as mg_compiler
from dask.delayed import delayed
from metagraph.tests.util import default_plugin_resolver, IdentityCompiler
from metagraph import translator, abstract_algorithm, concrete_algorithm
import networkx as nx
from metagraph.core.resolver import Resolver
from metagraph.core.dask.resolver import DaskResolver
from metagraph import PluginRegistry
from pytest import fixture
import pytest
import numpy as np
import dask


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

    subgraphs = mg_compiler.extract_compilable_subgraphs(
        z.dask, output_keys=[z.key], compiler="noexist"
    )
    assert len(subgraphs) == 0


def test_extract_subgraphs_singleton(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z = scale_func(a, 2.0)

    # default behavior is to include compilable single node subgraphs
    subgraphs = mg_compiler.extract_compilable_subgraphs(
        z.__dask_graph__(), output_keys=[z.key], compiler="identity_comp"
    )
    assert len(subgraphs) == 1

    # disable
    subgraphs = mg_compiler.extract_compilable_subgraphs(
        z._dsk, compiler="identity_comp", output_keys=[z.key], include_singletons=False
    )
    assert len(subgraphs) == 0


def test_extract_subgraphs_chain(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z = scale_func(scale_func(scale_func(a, 2.0), 3.0), 4.0)

    subgraphs = mg_compiler.extract_compilable_subgraphs(
        z.__dask_graph__(), output_keys=[z.key], compiler="identity_comp"
    )
    assert len(subgraphs) == 1
    subgraph = subgraphs[0]
    assert len(subgraph.tasks) == 3
    # FIXME: This is zero because the input numpy array is not wrapped in its own placeholder object
    assert len(subgraph.input_keys) == 0
    assert subgraph.output_key == z.key


def test_extract_subgraphs_two_chains(res):
    """Two chains feeding into a combining node"""
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z1 = scale_func(scale_func(scale_func(a, 2.0), 3.0), 4.0)
    z2 = scale_func(scale_func(scale_func(a, 2.5), 3.5), 4.5)
    merge = res.algos.testing.add(z1, z2)

    # The merge node cannot be fused with z1 or z2 without reducing parallelism in the graph

    subgraphs = mg_compiler.extract_compilable_subgraphs(
        merge.__dask_graph__(),
        compiler="identity_comp",
        output_keys=[merge.key],
        include_singletons=False,  # exclude the add node
    )
    assert len(subgraphs) == 2
    for subgraph in subgraphs:
        assert len(subgraph.tasks) == 3
        # FIXME: This is zero because the input numpy array is not wrapped in its own placeholder object
        assert len(subgraph.input_keys) == 0
        # we don't know what order the two chains will come out in
        assert subgraph.output_key in (z1.key, z2.key)

    # now check if we get the add node
    subgraphs = mg_compiler.extract_compilable_subgraphs(
        merge.__dask_graph__(),
        output_keys=[merge.key],
        compiler="identity_comp",
        include_singletons=True,
    )
    assert len(subgraphs) == 3
    assert merge.key in [s.output_key for s in subgraphs]


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
        ans.__dask_graph__(), output_keys=[ans.key], compiler="identity_comp"
    )
    assert len(subgraphs) == 3

    # separate the final chain from the input chains
    final_chain = None
    input_chains = []
    for subgraph in subgraphs:
        # FIXME: key property
        if subgraph.output_key == ans.key:
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
        assert input_key in (z1.key, z2.key)
    assert (
        final_chain.output_key == ans.key
    )  # true by construction, checked here for completeness

    # input_chain tests
    for subgraph in input_chains:
        assert len(subgraph.tasks) == 3
        # FIXME: This is zero because the input numpy array is not wrapped in its own placeholder object
        assert len(subgraph.input_keys) == 0
        # we don't know what order the two chains will come out in
        # FIXME: key property
        assert subgraph.output_key in (z1.key, z2.key)


def test_extract_subgraphs_diamond(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    top_node = res.algos.testing.offset(a, offset=2.0)
    left_node = scale_func(top_node, 3.0)
    right_node = scale_func(top_node, 5.0)
    bottom_node = res.algos.testing.add(left_node, right_node)
    result_node = bottom_node

    subgraphs = mg_compiler.extract_compilable_subgraphs(
        result_node.__dask_graph__(),
        output_keys=[result_node.key],
        compiler="identity_comp",
    )
    assert len(subgraphs) == 4

    key_to_node = {
        top_node.key: top_node,
        left_node.key: left_node,
        right_node.key: right_node,
        bottom_node.key: bottom_node,
    }
    node_key_to_input_node_keys = {
        top_node.key: set(),
        left_node.key: {top_node.key},
        right_node.key: {top_node.key},
        bottom_node.key: {left_node.key, right_node.key},
    }

    for subgraph in subgraphs:
        expected_input_node_keys = node_key_to_input_node_keys[subgraph.output_key]
        assert set(subgraph.input_keys) == expected_input_node_keys

    subgraphs = mg_compiler.extract_compilable_subgraphs(
        result_node.__dask_graph__(),
        compiler="identity_comp",
        output_keys=[result_node.key],
        include_singletons=False,
    )
    assert len(subgraphs) == 0


def test_compile_subgraphs_noop(res):
    a = res.wrappers.NodeSet.NumpyNodeSet(np.arange(100))

    compiler = res.compilers["identity_comp"]

    optimized_dsk = mg_compiler.compile_subgraphs(
        a.__dask_graph__(), output_keys=[a.key], compiler=compiler
    )
    assert len(optimized_dsk) == 1
    assert a.key in optimized_dsk


def test_compile_subgraphs_three_chains(res):
    """Compile Y-shaped graph"""
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z1 = scale_func(scale_func(scale_func(a, 2.0), 3.0), 4.0)
    z2 = scale_func(scale_func(scale_func(a, 2.5), 3.5), 4.5)
    merge = res.algos.testing.add(z1, z2)
    ans = scale_func(merge, 2.8)

    compiler = res.compilers["identity_comp"]

    optimized_dsk = mg_compiler.compile_subgraphs(
        ans.__dask_graph__(), output_keys=[ans.key], compiler=compiler
    )
    assert len(optimized_dsk) == 3
    assert z1.key in optimized_dsk
    assert z2.key in optimized_dsk
    assert ans.key in optimized_dsk

    optimized_result = dask.core.get(optimized_dsk, ans.key)
    unoptimized_result = ans.compute(optimize_graph=False)
    numpy_result = 2.8 * ((a * 2 * 3 * 4) + (a * 2.5 * 3.5 * 4.5))
    np.testing.assert_array_equal(optimized_result, numpy_result)
    np.testing.assert_array_equal(unoptimized_result, numpy_result)


def test_compile_subgraph_kwargs(res):
    """Compile subgraph with task that has kwargs"""
    a = np.arange(100)
    offset_func = res.algos.testing.offset
    z = offset_func(offset_func(a=a, offset=1.0), offset=2.0)

    compiler = res.compilers["identity_comp"]

    optimized_dsk = mg_compiler.compile_subgraphs(
        z.__dask_graph__(), output_keys=[z.key], compiler=compiler
    )
    assert len(optimized_dsk) == 1

    optimized_result = dask.core.get(optimized_dsk, z.key)
    unoptimized_result = z.compute(optimize_graph=False)
    numpy_result = a + 1 + 2
    np.testing.assert_array_equal(optimized_result, numpy_result)
    np.testing.assert_array_equal(unoptimized_result, numpy_result)


def test_extract_subgraphs_multiple_outputs(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    x = scale_func(a, 2.0)
    y = scale_func(x, 3.0)
    z = scale_func(y, 4.0)

    subgraphs = mg_compiler.extract_compilable_subgraphs(
        z.__dask_graph__(), output_keys=[z.key, y.key], compiler="identity_comp"
    )
    assert len(subgraphs) == 2
    for subgraph in subgraphs:
        if subgraph.output_key == z.key:
            assert len(subgraph.tasks) == 1
            assert subgraph.input_keys == [y.key]
        elif subgraph.output_key == y.key:
            assert len(subgraph.tasks) == 2
            # FIXME: This is zero because the input numpy array is not wrapped in its own placeholder object
            assert subgraph.input_keys == []
        else:
            assert Fail, f"unexpected subgraph with output key {subgraph.output_key}"


def test_compile_subgraphs_multiple_outputs(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    x = scale_func(a, 2.0)
    y = scale_func(x, 3.0)
    z = scale_func(y, 4.0)

    compiler = res.compilers["identity_comp"]
    optimized_dsk = mg_compiler.compile_subgraphs(
        z.__dask_graph__(), output_keys=[z.key, y.key], compiler=compiler
    )
    assert len(optimized_dsk) == 2
    z_comp, y_comp = dask.core.get(optimized_dsk, [z.key, y.key])
    np.testing.assert_array_equal(z_comp, a * 2 * 3 * 4)
    np.testing.assert_array_equal(y_comp, a * 2 * 3)


def test_optimize(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    x = scale_func(a, 2.0)
    y = scale_func(x, 3.0)
    z = scale_func(y, 4.0)

    compiler = res.compilers["identity_comp"]
    optimized_dsk = mg_compiler.optimize(
        z.__dask_graph__(), output_keys=[z.key, y.key], compiler=compiler
    )
    assert len(optimized_dsk) == 2


def test_optimize_cull(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    z1 = scale_func(scale_func(scale_func(a, 2.0), 3.0), 4.0)
    z2 = scale_func(scale_func(scale_func(a, 2.5), 3.5), 4.5)
    merge = res.algos.testing.add(z1, z2)
    ans = scale_func(merge, 2.8)

    compiler = res.compilers["identity_comp"]
    optimized_dsk = mg_compiler.optimize(
        ans.__dask_graph__(), output_keys=[z2.key], compiler=compiler
    )
    assert len(optimized_dsk) == 1


def test_automatic_optimize(res):
    a = np.arange(100)
    scale_func = res.algos.testing.scale
    x = scale_func(a, 2.0)
    y = scale_func(x, 3.0)
    z = scale_func(y, 4.0)

    compiler = res.compilers["identity_comp"]

    # expect 1 compiled chain
    compiler.clear_trace()
    np.testing.assert_array_equal(z.compute(), a * 2 * 3 * 4)
    assert len(compiler.compile_subgraph_calls) == 1

    # expect 2 compiled chains
    compiler.clear_trace()
    result = dask.compute(z, y)
    np.testing.assert_array_equal(result[0], a * 2 * 3 * 4)
    np.testing.assert_array_equal(result[1], a * 2 * 3)
    assert len(compiler.compile_subgraph_calls) == 2

    # expect no compiled chains
    compiler.clear_trace()
    np.testing.assert_array_equal(z.compute(optimize_graph=False), a * 2 * 3 * 4)
    assert len(compiler.compile_subgraph_calls) == 0


def test_dfs_bug():
    from metagraph.core.compiler import _dfs_sorted_dask_keys
    from collections import defaultdict

    compilable_keys = {
        ("call-a79434332c1962a6347ce169a566b3ed", "ex.relu"),
        ("call-585a8a06d235f777867e08955d490071", "ex.fully_connected_layer"),
        ("call-7da00fd444fc128ceacb194485c1ff15", "ex.neighbor_features"),
    }

    dependencies = {
        ("call-a79434332c1962a6347ce169a566b3ed", "ex.relu"): {
            ("call-585a8a06d235f777867e08955d490071", "ex.fully_connected_layer")
        },
        ("call-585a8a06d235f777867e08955d490071", "ex.fully_connected_layer"): {
            ("call-7da00fd444fc128ceacb194485c1ff15", "ex.neighbor_features")
        },
        ("call-7da00fd444fc128ceacb194485c1ff15", "ex.neighbor_features"): {
            (
                "translate-f7233ee614fb487c5360be7d042cee54",
                "ScipyGraphType->MLIRGraphBLASGraphType",
            )
        },
        (
            "translate-f7233ee614fb487c5360be7d042cee54",
            "ScipyGraphType->MLIRGraphBLASGraphType",
        ): set(),
    }

    dependents = defaultdict(
        None,
        {
            ("call-a79434332c1962a6347ce169a566b3ed", "ex.relu"): set(),
            ("call-585a8a06d235f777867e08955d490071", "ex.fully_connected_layer"): {
                ("call-a79434332c1962a6347ce169a566b3ed", "ex.relu")
            },
            ("call-7da00fd444fc128ceacb194485c1ff15", "ex.neighbor_features"): {
                ("call-585a8a06d235f777867e08955d490071", "ex.fully_connected_layer")
            },
            (
                "translate-f7233ee614fb487c5360be7d042cee54",
                "ScipyGraphType->MLIRGraphBLASGraphType",
            ): {("call-7da00fd444fc128ceacb194485c1ff15", "ex.neighbor_features")},
        },
    )

    ordered_keys = list(
        _dfs_sorted_dask_keys(compilable_keys, dependencies, dependents)
    )

    assert len(ordered_keys) == 3


@fixture
def res():
    from metagraph.plugins.core.types import Vector
    from metagraph.plugins.numpy.types import NumpyVectorType

    @abstract_algorithm("testing.add")
    def testing_add(a: Vector, b: Vector) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.add", compiler="identity_comp")
    def compiled_add(a: NumpyVectorType, b: NumpyVectorType) -> NumpyVectorType:
        return a + b

    @abstract_algorithm("testing.scale")
    def testing_scale(a: Vector, scale: float) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.scale", compiler="identity_comp")
    def compiled_scale(a: NumpyVectorType, scale: float) -> NumpyVectorType:
        return a * scale

    @abstract_algorithm("testing.offset")
    def testing_offset(a: Vector, *, offset: float) -> Vector:  # pragma: no cover
        pass

    @concrete_algorithm("testing.offset", compiler="identity_comp")
    def compiled_offset(a: NumpyVectorType, *, offset: float) -> NumpyVectorType:
        return a + offset

    registry = PluginRegistry("test_subgraphs_plugin")
    registry.register(testing_add)
    registry.register(compiled_add)
    registry.register(testing_scale)
    registry.register(compiled_scale)
    registry.register(testing_offset)
    registry.register(compiled_offset)
    registry.register(IdentityCompiler())

    resolver = Resolver()
    resolver.load_plugins_from_environment()
    resolver.register(registry.plugins)

    return DaskResolver(resolver)
