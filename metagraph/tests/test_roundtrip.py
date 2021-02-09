import pytest
from metagraph.core.roundtrip import RoundTripper, UnreachableTranslationError
from metagraph.core.resolver import Resolver
from metagraph.core.dask.resolver import DaskResolver
from metagraph.core.plugin_registry import PluginRegistry
from .util import default_plugin_resolver
import grblas


def test_roundtripper(default_plugin_resolver):
    dpr = default_plugin_resolver

    # Register translators that aren't round-trippable to induce failures
    registry = PluginRegistry("test_roundtripper")
    for at in dpr.abstract_types:
        registry.register(at)
    for ct in dpr.concrete_types:
        registry.register(ct)
    registry.register(
        dpr.translators[
            (dpr.types.NodeMap.PythonNodeMapType, dpr.types.NodeMap.NumpyNodeMapType)
        ]
    )
    registry.register(
        dpr.translators[
            (dpr.types.NodeMap.NumpyNodeMapType, dpr.types.NodeMap.GrblasNodeMapType)
        ]
    )
    registry.register(
        dpr.translators[
            (dpr.types.NodeMap.PythonNodeMapType, dpr.types.NodeSet.PythonNodeSetType)
        ]
    )

    res = Resolver()
    res.register(registry.plugins)
    # If test is in dask mode, convert our resolver to a dask resolver
    if isinstance(dpr, DaskResolver):
        res = DaskResolver(res)

    gnm = dpr.wrappers.NodeMap.GrblasNodeMap(
        grblas.Vector.from_values([0, 2], [1.1, 5.5])
    )
    gns = dpr.wrappers.NodeSet.GrblasNodeSet(grblas.Vector.from_values([0, 2], [1, 1]))
    pnm = {0: 1.1, 2: 5.5}
    pns = {0, 2}

    rt = RoundTripper(res)
    with pytest.raises(
        UnreachableTranslationError, match="Impossible to return from target"
    ):
        rt.verify_round_trip(pnm)
    with pytest.raises(UnreachableTranslationError, match="Impossible to reach source"):
        rt.verify_round_trip(gnm)

    with pytest.raises(
        TypeError, match="start and end must have different abstract types"
    ):
        rt.verify_one_way(pnm, gnm)
    with pytest.raises(UnreachableTranslationError, match="Impossible to reach source"):
        rt.verify_one_way(gnm, pns)
    with pytest.raises(
        UnreachableTranslationError, match="Impossible to return from target"
    ):
        rt.verify_one_way(pnm, gns)
