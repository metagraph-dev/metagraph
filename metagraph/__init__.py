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
from .core.node_labels import NodeLabels
from .core.typing import Union, Optional, List
from .types import NodeID
from . import types, algorithms

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

_SPECIAL_ATTRS = ["resolver", "algo", "translate", "type_of", "typeclass_of"]


def __getattr__(name):
    """Lazy load the global resolver to avoid circular dependencies with plugins."""

    if name in _SPECIAL_ATTRS:
        from .core.resolver import Resolver

        res = Resolver()
        res.load_plugins_from_environment()
        globals()["resolver"] = res
        globals()["algos"] = res.algos
        globals()["translate"] = res.translate
        globals()["type_of"] = res.type_of
        globals()["typeclass_of"] = res.typeclass_of

        return globals()[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    attrs = list(globals().keys())
    if "resolver" not in attrs:
        attrs += _SPECIAL_ATTRS
    return attrs
