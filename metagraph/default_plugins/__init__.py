from ..core.plugin_registry import PluginRegistry

# Use this as the entry_point object
registry = PluginRegistry()


def find_plugins():
    # Ensure we import all items we want registered
    from . import abstract_types, wrappers, algorithms, algorithms

    return registry.plugins
