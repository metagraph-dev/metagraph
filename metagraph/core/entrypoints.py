"""Plugin discovery using setuptools entrypoints mechanism

In setup.py, use something like:

setup(
    ...,
    entry_points={
        "metagraph.plugins": [
            "plugins" = mypackage.mysubmodule:find_plugins()"
        ],
    },
)

The find_plugins function should return a dict of lists of the appropriate classes
(types) or objects (translators and functions).
Allowable keys in the plugins dict:
 - abstract_types
 - concrete_types
 - wrappers
 - translators
 - abstract_algorithms
 - concrete_algorithms
"""

from typing import List, Dict, Callable
import importlib_metadata


class EntryPointsError(Exception):
    pass


def load_plugins() -> Dict[str, List]:
    """Return a list of plugins of particular kind.

    See find_plugin_loaders() for valid kind values.
    """
    plugin_loaders = importlib_metadata.entry_points().get("metagraph.plugins", [])
    plugins = {}
    for pl in plugin_loaders:
        if pl.name != "plugins":
            raise EntryPointsError(
                f"metagraph.plugin found an unexpected entry_point: {pl.name}"
            )
        else:
            plugin_loader = pl.load()
            plugins.update(plugin_loader())

    # Convenience feature for developers of metagraph
    # If default_plugins aren't loaded (because metagraph isn't actually installed), load them now
    import metagraph as mg

    if not hasattr(mg, "default_plugins"):
        from metagraph import default_plugins

        plugins.update(default_plugins.find_plugins())

    return plugins
