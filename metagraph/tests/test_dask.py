import pytest
from metagraph.core.resolver import Resolver
from metagraph.core import dask as mgdask
from metagraph.tests.util import default_plugin_resolver
from metagraph.plugins.python.types import PythonNodeMap
from metagraph.plugins.numpy.types import NumpyNodeMap
from metagraph.plugins.graphblas.types import GrblasNodeMap
import grblas


def test_translation_direct(default_plugin_resolver):
    dpr = default_plugin_resolver
    ldpr = mgdask.DaskResolver(dpr)
    x = PythonNodeMap({0: 12.5, 1: 33.4, 42: -1.2})
    final = GrblasNodeMap(
        grblas.Vector.from_values([0, 1, 42], [12.5, 33.4, -1.2], size=43),
    )
    y = ldpr.translate(x, NumpyNodeMap)
    z = ldpr.translate(y, GrblasNodeMap)
    assert isinstance(y, mgdask.placeholder.Placeholder)
    assert isinstance(z, mgdask.placeholder.Placeholder)
    assert len(y._dsk.keys()) == 1  # Only one task to perform
    assert len(z._dsk.keys()) == 2  # Two tasks to perform because y is still lazy
    dpr.assert_equal(z.compute(), final)


def test_translation_multistep(default_plugin_resolver):
    dpr = default_plugin_resolver
    res_small = Resolver()
    # Only register some of the translators to force a multi-step translation path
    res_small.register(
        {
            "foo": {
                "abstract_types": dpr.abstract_types,
                "concrete_types": dpr.concrete_types,
                "wrappers": {PythonNodeMap, NumpyNodeMap, GrblasNodeMap},
                "translators": {
                    dpr.translators[(PythonNodeMap.Type, NumpyNodeMap.Type)],
                    dpr.translators[(NumpyNodeMap.Type, GrblasNodeMap.Type)],
                },
            }
        }
    )
    ldpr = mgdask.DaskResolver(res_small)
    x = PythonNodeMap({0: 12.5, 1: 33.4, 42: -1.2})
    final = GrblasNodeMap(
        grblas.Vector.from_values([0, 1, 42], [12.5, 33.4, -1.2], size=43),
    )
    z = ldpr.translate(x, GrblasNodeMap)
    assert isinstance(z, mgdask.placeholder.Placeholder)
    assert len(z._dsk.keys()) == 2  # Only one translation, but creates two tasks
    dpr.assert_equal(z.compute(), final)
