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

from typing import Set, Dict, Callable
import importlib_metadata


class EntryPointsError(Exception):
    pass


def load_plugins() -> Dict[str, Set]:
    """Return a list of plugins of particular kind.

    See find_plugin_loaders() for valid kind values.
    """
    plugin_loaders = importlib_metadata.entry_points().get("metagraph.plugins", [])
    plugins = {}
    seen = set()
    for pl in plugin_loaders:
        if pl.name != "plugins":
            raise EntryPointsError(
                f"metagraph.plugin found an unexpected entry_point: {pl.name}"
            )
        elif pl not in seen:
            plugin_loader = pl.load()

            plugin_items = plugin_loader()
            for key, vals in plugin_items.items():
                if key not in plugins:
                    plugins[key] = set(vals)
                else:
                    plugins[key].update(vals)
            seen.add(pl)
    return plugins
