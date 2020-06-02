import pytest
import metagraph as mg
from metagraph import PluginRegistry
from metagraph.core.plugin_registry import PluginRegistryError
from .site_dir import plugin1


def test_registry_modules():
    reg = PluginRegistry("test_registry_modules_default_plugin")
    reg.register_from_modules(mg.types, mg.algorithms)
    reg.register_from_modules(plugin1, name="plugin1")
    plugins = plugin1.find_plugins()
    assert len(reg.plugins["plugin1"]["abstract_types"]) == len(
        plugins["plugin1"]["abstract_types"]
    )
    assert len(reg.plugins["plugin1"]["abstract_algorithms"]) == len(
        plugins["plugin1"]["abstract_algorithms"]
    )
    assert len(reg.plugins["plugin1"]["concrete_types"]) == len(
        plugins["plugin1"]["concrete_types"]
    )
    assert len(reg.plugins["plugin1"]["wrappers"]) == len(
        plugins["plugin1"]["wrappers"]
    )
    assert len(reg.plugins["plugin1"]["translators"]) == len(
        plugins["plugin1"]["translators"]
    )
    assert len(reg.plugins["plugin1"]["concrete_algorithms"]) == len(
        plugins["plugin1"]["concrete_algorithms"]
    )

    assert len(reg.plugins["plugin1"]["abstract_types"]) > 0
    assert len(reg.plugins["plugin1"]["abstract_algorithms"]) > 0
    assert len(reg.plugins["plugin1"]["concrete_types"]) > 0
    assert len(reg.plugins["plugin1"]["wrappers"]) > 0
    assert len(reg.plugins["plugin1"]["translators"]) > 0
    assert len(reg.plugins["plugin1"]["concrete_algorithms"]) > 0

    with pytest.raises(
        TypeError,
        match="Expected one or more modules. Got a type <class 'int'> instead.",
    ):
        reg.register_from_modules([7], name="bad_plugin")


def test_registry_failures():
    reg = PluginRegistry("test_registry_failures_default_plugin")

    with pytest.raises(PluginRegistryError, match="Invalid type for plugin registry"):

        class NotValid:
            pass

        reg.register(NotValid, "invalid_plugin")

    with pytest.raises(PluginRegistryError, match="Invalid object for plugin registry"):

        def not_valid():  # pragma: no cover
            pass

        reg.register(not_valid, "invalid_plugin")
