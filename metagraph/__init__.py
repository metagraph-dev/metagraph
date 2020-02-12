from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# Public API
from .core.plugin import (
    AbstractType,
    ConcreteType,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)
from .core import dtypes
from .core.plugin_registry import PluginRegistry
