import pytest
import metagraph as mg
from metagraph import PluginRegistry
from metagraph.core.plugin_registry import PluginRegistryError
from .site_dir import plugin1


def test_registry_modules():
    reg = PluginRegistry()
    reg.register_from_modules(plugin1)
    plugins = plugin1.find_plugins()
    assert len(reg.plugins["abstract_types"]) == len(plugins["abstract_types"])
    assert len(reg.plugins["concrete_types"]) == len(plugins["concrete_types"])
    assert len(reg.plugins["wrappers"]) == len(plugins["wrappers"])
    assert len(reg.plugins["translators"]) == len(plugins["translators"])
    assert len(reg.plugins["abstract_algorithms"]) == len(
        plugins["abstract_algorithms"]
    )
    assert len(reg.plugins["concrete_algorithms"]) == len(
        plugins["concrete_algorithms"]
    )

    # List also allowed
    reg2 = PluginRegistry()
    reg2.register_from_modules([plugin1])
    assert reg2.plugins == reg.plugins

    with pytest.raises(
        TypeError,
        match="Expected one or more modules.  Got a type <class 'int'> instead",
    ):
        reg2.register_from_modules(7)


def test_registry_failures():
    reg = PluginRegistry()

    with pytest.raises(PluginRegistryError, match="Invalid type for plugin registry"):

        @reg.register
        class NotValid:
            pass

    with pytest.raises(PluginRegistryError, match="Invalid object for plugin registry"):

        @reg.register
        def not_valid():  # pragma: no cover
            pass
