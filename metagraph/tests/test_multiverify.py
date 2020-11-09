import pytest

grblas = pytest.importorskip("grblas")

from metagraph.core.resolver import Resolver
from metagraph.core.multiverify import (
    MultiVerify,
    MultiResult,
    UnsatisfiableAlgorithmError,
)
from .util import default_plugin_resolver
import numpy as np
from metagraph.plugins.graphblas.types import GrblasNodeMap
from metagraph.plugins.numpy.types import NumpyNodeMap
from metagraph.plugins.python.types import PythonNodeMapType


def test_multiresult_consistent_length(default_plugin_resolver):
    mv = MultiVerify(default_plugin_resolver)
    mr = MultiResult(mv, {"foo.bar": (1, 2, "c"), "foo.bar2": (3, 4, "f"),})

    with pytest.raises(ValueError, match="length mismatch"):
        MultiResult(mv, {"foo.bar": (1, 2, 3), "foo.bar2": (1, 2, 3, 4),})

    with pytest.raises(ValueError, match="length mismatch"):
        MultiResult(mv, {"foo.tuple": (1,), "foo.scalar": 1,})


def test_multiresult_getitem(default_plugin_resolver):
    mv = MultiVerify(default_plugin_resolver)
    mr_single = MultiResult(mv, {"a": 1, "b": 2})
    mr_multi = MultiResult(mv, {"a": (1, 2, 3), "b": (2, 3, 4)})

    with pytest.raises(TypeError, match="Results are not multi-valued"):
        mr_single[0]

    mr2_3 = mr_multi[1:]
    assert mr2_3._results["a"] == (2, 3)


def test_multiresult_normalize(default_plugin_resolver):
    dpr = default_plugin_resolver
    mv = MultiVerify(dpr)
    nnm = dpr.wrappers.NodeMap.NumpyNodeMap(
        np.array([1.1, 5.5]), nodes=np.array([0, 2])
    )
    pnm = {0: 1.1, 2: 5.5}

    mr = MultiResult(mv, {"testing.nnm": (nnm, 1), "testing.pnm": (pnm, 2)})
    assert not mr._normalized
    norm1 = mr.normalize((dpr.wrappers.NodeMap.GrblasNodeMap, float))
    assert norm1._normalized
    assert set(norm1._results.keys()) == {"testing.nnm", "testing.pnm"}
    assert type(norm1._results["testing.nnm"][0]) is GrblasNodeMap
    assert type(norm1._results["testing.pnm"][0]) is GrblasNodeMap
    assert (
        type(norm1._results["testing.pnm"][1]) is int
    )  # anything not a concrete type is ignored (ex. float)

    # Passing None to normalize ignores that argument
    norm2 = mr.normalize((None, None))
    assert type(norm2._results["testing.nnm"][0]) is NumpyNodeMap
    assert type(norm2._results["testing.pnm"][0]) is dict  #  PythonNodeMapType
    assert type(norm2._results["testing.pnm"][1]) is int

    with pytest.raises(
        TypeError,
        match="Cannot normalize results of length 2 into something of length 3",
    ):
        mr.normalize((GrblasNodeMap, float, float))

    with pytest.raises(
        TypeError,
        match="Cannot normalize results of length 2 into something of length None",
    ):
        mr.normalize(GrblasNodeMap)

    with pytest.raises(
        TypeError,
        match="Cannot normalize results of length None into something of length 2",
    ):
        mr[0].normalize((GrblasNodeMap, float))


def test_unnormalizable(default_plugin_resolver):
    dpr = default_plugin_resolver
    # Only register some of the translators to force a multi-step translation path
    res_small = Resolver()
    res_small.register(
        {
            "foo": {
                "abstract_types": dpr.abstract_types,
                "concrete_types": dpr.concrete_types,
                "wrappers": {PythonNodeMapType, NumpyNodeMap, GrblasNodeMap},
                "translators": {
                    dpr.translators[(PythonNodeMapType, NumpyNodeMap.Type)],
                },
            }
        }
    )
    mv = MultiVerify(res_small)
    mr = MultiResult(mv, {"testing.foo": {0: 1.1, 2: 4.5}})

    with pytest.raises(UnsatisfiableAlgorithmError, match="Unable to convert type"):
        mr.normalize(GrblasNodeMap)


def test_compute(default_plugin_resolver):
    dpr = default_plugin_resolver
    mv = MultiVerify(dpr)
    mr = MultiResult(mv, {"testing.foo": 1, "testing.bar": 2,})

    with pytest.raises(
        TypeError, match='"algo" must be of type `str` or `Dispatcher`, not'
    ):
        mv.compute(14, 15, foo=True)

    with pytest.raises(
        TypeError, match='Invalid call signature for "util.nodemap.select":'
    ):
        mv.compute("util.nodemap.select", 1, 2, 3, 4, foo=True)

    with pytest.raises(
        TypeError, match='Invalid argument "nodes"; may not be a MultiResult'
    ):
        mv.compute("util.nodemap.select", None, mr)

    with pytest.raises(UnsatisfiableAlgorithmError, match="No plan found"):
        mv.compute("util.nodemap.select", 1, 2)


def test_transform(default_plugin_resolver):
    dpr = default_plugin_resolver
    mv = MultiVerify(dpr)

    with pytest.raises(TypeError, match="Cannot find"):
        mv.transform("testing.foo.bar", 1, 2, foo=True)

    with pytest.raises(
        TypeError, match='"algo" must be of type `str` or `ExactDispatcher`'
    ):
        mv.transform(14, 15, foo=True)

    with pytest.raises(TypeError, match="requires at least one MultiResult argument"):
        mv.transform("util.nodemap.select.core_numpy", 1, 2)

    mr_unnormalized = MultiResult(mv, {"testing.foo": 1, "testing.bar": 2})

    with pytest.raises(TypeError, match="must be normalized"):
        mv.transform(
            dpr.algos.util.nodemap.select.core_numpy, mr_unnormalized, mr_unnormalized
        )

    mr_empty = MultiResult(mv, {}, normalized=True)

    with pytest.raises(ValueError, match="has no results"):
        mv.transform(
            dpr.plugins.core_numpy.algos.util.nodemap.select, mr_empty, mr_empty
        )


def test_transform_combos(default_plugin_resolver):
    dpr = default_plugin_resolver
    mv = MultiVerify(dpr)
    pnm = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    pns = {0, 2, 4}
    nodemaps = MultiResult(
        mv,
        {"testing.foo": pnm, "testing.bar": pnm, "testing.baz": pnm},
        normalized=True,
    )
    nodesets = MultiResult(
        mv, {"testing.spam": pns, "testing.eggs": pns}, normalized=True
    )

    result = mv.transform(dpr.algos.util.nodemap.select.core_python, nodemaps, nodesets)
    assert len(result._results) == len(nodemaps._results) * len(nodesets._results)


def test_custom_compare(default_plugin_resolver):
    dpr = default_plugin_resolver
    mv = MultiVerify(dpr)
    mr = MultiResult(mv, {"testing.foo": 10})

    def cmp_func(result):
        assert result == 5

    with pytest.raises(AssertionError):
        mv.custom_compare(mr, cmp_func=cmp_func)

    mr2 = MultiResult(mv, {"testing.bar": {0, 2, 4, 7}})
    mr2 = mr2.normalize(None)  # force computation for lazy objects

    def cmp_func2(result):
        assert result == {0, 2, 4, 100}

    with pytest.raises(AssertionError):
        mv.custom_compare(mr2, cmp_func2)


def test_compare_values(default_plugin_resolver, capsys):
    dpr = default_plugin_resolver
    mv = MultiVerify(dpr)
    pnm1 = {0: 1.1, 2: 4.5}
    pnm2 = {0: 1.1, 2: 4.9}
    pns = {0, 2}

    with pytest.raises(TypeError, match="`val` must be"):
        mv.compare_values(pns, pnm1, "testing.compare.values")

    capsys.readouterr()
    with pytest.raises(AssertionError):
        mv.compare_values(pnm1, pnm2, "testing.compare.values")
    captured = capsys.readouterr()
    assert "val" in captured.out
    assert "expected_val" in captured.out

    # Compare floats
    mv.compare_values(1.1, 1.1 + 1e-9, "testing.compare.values.floats", rel_tol=1e-6)

    # Compare ints
    mv.compare_values(5, 5, "testing.compare.values.ints")
