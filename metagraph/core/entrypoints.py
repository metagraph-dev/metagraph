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

The find_plugins function should return a metagraph.PluginRegistry of the relevant plugins.
"""

from typing import Set, Dict, Callable
import importlib_metadata
from .plugin_registry import PluginRegistry


class EntryPointsError(Exception):
    pass


def load_plugins() -> PluginRegistry:
    entry_points = importlib_metadata.entry_points().get("metagraph.plugins", [])
    combined_registry = PluginRegistry()
    seen = set()
    for entry_point in entry_points:
        if entry_point.name != "plugins":
            raise EntryPointsError(
                f"metagraph.plugin found an unexpected entry_point: {entry_point.name}"
            )
        elif entry_point not in seen:
            plugin_loader = entry_point.load()

            current_registry = plugin_loader()
            combined_registry.update(current_registry)
            seen.add(entry_point)

    return combined_registry
