from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# Public API
from .core.plugin import (
    AbstractType,
    ConcreteType,
    Wrapper,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)
from .core import dtypes
from .core.plugin_registry import PluginRegistry
from .core.node_index import IndexedNodes, SequentialNodes
from . import types, algorithms

_SPECIAL_ATTRS = ["resolver", "algo", "translate", "typeof"]


def __getattr__(name):
    """Lazy load the global resolver to avoid circular dependencies with plugins."""

    if name in _SPECIAL_ATTRS:
        from .core.resolver import Resolver

        res = Resolver()
        res.load_plugins_from_environment()
        globals()["resolver"] = res
        globals()["algo"] = res.algo
        globals()["translate"] = res.translate
        globals()["typeof"] = res.typeof

        return globals()[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    attrs = list(globals().keys())
    if "resolver" not in attrs:
        attrs += _SPECIAL_ATTRS
    return attrs
