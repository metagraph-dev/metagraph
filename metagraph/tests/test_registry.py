import pytest
import metagraph as mg
from metagraph import PluginRegistry
from metagraph.core.plugin_registry import PluginRegistryError
from .site_dir import plugin1_util, plugin1


def test_registry_modules():
    reg = PluginRegistry()
    reg.register_from_modules(
        "seeing_this_plugin_name_indicates_bug", [mg.types, mg.algorithms]
    )
    reg.register_from_modules("plugin1", [plugin1_util])
    plugins = plugin1.find_plugins()
    assert len(reg.abstract_types) == len(plugins.abstract_types)
    assert len(reg.abstract_algorithms) == len(plugins.abstract_algorithms)
    assert len(reg.concrete_types) == len(plugins.concrete_types)
    assert len(reg.wrappers) == len(plugins.wrappers)
    assert len(reg.translators) == len(plugins.translators)
    assert len(reg.concrete_algorithms) == len(plugins.concrete_algorithms)

    with pytest.raises(
        TypeError,
        match="Expected one or more modules.  Got a type <class 'int'> instead",
    ):
        reg.register_from_modules("bad_plugin", [7])


def test_registry_failures():
    reg = PluginRegistry()

    with pytest.raises(
        PluginRegistryError, match="Invalid abstract type for plugin registry"
    ):

        @reg.register_abstract
        class NotValid:
            pass

    with pytest.raises(
        PluginRegistryError, match="Invalid concrete type for plugin registry"
    ):

        class NotValid:
            pass

        reg.register_concrete("bad_class_plugin", NotValid)

    with pytest.raises(
        PluginRegistryError, match="Invalid abstract object for plugin registry"
    ):

        @reg.register_abstract
        def not_valid():  # pragma: no cover
            pass

    with pytest.raises(
        PluginRegistryError, match="Invalid concrete object for plugin registry"
    ):

        def not_valid():  # pragma: no cover
            pass

        reg.register_concrete("bad_func_plugin", not_valid)
