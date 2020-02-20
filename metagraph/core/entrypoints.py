"""Plugin discovery using setuptools entrypoints mechanism

In setup.py, use something like:

setup(
    ...,
    entry_points={
        "metagraph.plugins": [
            "abstract_types = mypackage.mysubmodule:abstract_types_plugin_func",
            "concrete_types = mypackage.mysubmodule:abstract_types_plugin_func",
            "translators = mypackage.mysubmodule:translators_plugin_func",
            "abstract_algorithms = mypackage.mysubmodule:abstract_algorithms_plugin_func",
            "concrete_algorithms = mypackage.mysubmodule:concrete_algorithms_plugin_func",
        ],
    },
)

Each of these plugin functions should return a list of the approprate classes
(types) or objects (translators and functions).
"""

from typing import List, Callable
import importlib_metadata


def find_plugin_loaders(kind: str) -> List[Callable]:
    """Return a list of plugin loading functions discovered via the
    metagraph.plugins entrypoint

    kind - one of the plugin types: abstract_type, concrete_type, translator,
           abstract_algorithms, concrete_algorithms
    """
    plugins = importlib_metadata.entry_points().get("metagraph.plugins", [])
    return [ep.load() for ep in plugins if ep.name == kind]


def load_plugins(kind: str) -> List:
    """Return a list of plugins of particular kind.

    See find_plugin_loaders() for valid kind values.
    """
    plugins = []
    for loader in find_plugin_loaders(kind):
        plugins.extend(loader())
    return plugins
