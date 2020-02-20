import pytest
import metagraph as mg
from metagraph import PluginRegistry
from .site_dir import plugin1


def test_registry_modules():
    reg = PluginRegistry()
    reg.register_from_modules(plugin1)
    assert len(reg.plugins["abstract_types"]) == len(plugin1.abstract_types())
    assert len(reg.plugins["concrete_types"]) == len(plugin1.concrete_types())
    assert len(reg.plugins["wrappers"]) == len(plugin1.wrappers())
    assert len(reg.plugins["translators"]) == len(plugin1.translators())
    assert len(reg.plugins["abstract_algorithms"]) == len(plugin1.abstract_algorithms())
    assert len(reg.plugins["concrete_algorithms"]) == len(plugin1.concrete_algorithms())

    # List also allowed
    reg2 = PluginRegistry()
    reg2.register_from_modules([plugin1])
    assert reg2.plugins == reg.plugins

    with pytest.raises(
        TypeError,
        match="Expected one or more modules.  Got a type <class 'int'> instead",
    ):
        reg2.register_from_modules(7)
