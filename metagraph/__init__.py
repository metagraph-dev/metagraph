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
    Compiler,
)
from .core import dtypes
from .core.plugin_registry import PluginRegistry
from .core.node_labels import NodeLabels
from .core.typing import Union, Optional, List, NodeID

### Initiaize configuration and defaults

import donfig
import yaml
import os.path

config = donfig.Config("metagraph")
defaults_fn = os.path.join(os.path.dirname(__file__), "metagraph.yaml")
with open(defaults_fn) as f:
    defaults = yaml.safe_load(f)

config.update_defaults(defaults)
config.ensure_file(source=defaults_fn, comment=True)

del f
del defaults
del defaults_fn


### Lazy loading of special attributes that require loading plugins

_SPECIAL_ATTRS = [
    "resolver",
    "types",
    "wrappers",
    "algos",
    "translate",
    "run",
    "type_of",
    "typeclass_of",
    "plan",
    "visualize",
    "optimize",
]


def __getattr__(name):
    """Lazy load the global resolver to avoid circular dependencies with plugins."""

    if name in _SPECIAL_ATTRS:
        from .core import resolver

        res = resolver.Resolver()
        res.load_plugins_from_environment()
        _set_default_resolver(res)

        return globals()[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    attrs = list(globals().keys())
    if "resolver" not in attrs:
        attrs += _SPECIAL_ATTRS
    return attrs


def _set_default_resolver(res):
    # Update mg.resolver to res
    # Point all special attrs to the versions from res
    for attr in _SPECIAL_ATTRS:
        if attr == "resolver":
            globals()[attr] = res
        elif attr == "visualize":
            from metagraph.core.dask.visualize import visualize

            globals()[attr] = visualize
        elif attr == "optimize":
            from metagraph.core.compiler import optimize

            globals()[attr] = optimize
        else:
            globals()[attr] = getattr(res, attr)
