from ..core.plugin_registry import PluginRegistry

# Use this as the entry_point object
registry = PluginRegistry("metagraph_core")

# Ensure we import all items we want registered
from . import abstract_types, concrete_types, algorithms, algorithms
