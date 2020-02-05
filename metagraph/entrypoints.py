from typing import List, Callable
import importlib_metadata


def find_plugin_loaders(kind: str) -> List[Callable]:
    """Return a list of plugin loading functions discovered via the metagraph.plugins entrypoint

    kind - one of the plugin types: abstract_type, concrete_type, translator
    """
    plugins = importlib_metadata.entry_points().get("metagraph.plugins", [])
    return [ep.load() for ep in plugins if ep.name == kind]
