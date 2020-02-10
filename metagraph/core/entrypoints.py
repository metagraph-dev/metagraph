from typing import List, Callable
import importlib_metadata


def find_plugin_loaders(kind: str) -> List[Callable]:
    """Return a list of plugin loading functions discovered via the metagraph.plugins entrypoint

    kind - one of the plugin types: abstract_type, concrete_type, translator
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
