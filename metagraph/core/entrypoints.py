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

# Check if importlib.metadata is available (new in py=3.8)
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata


class EntryPointsError(Exception):
    pass


def load_plugins():
    entry_points = metadata.entry_points().get("metagraph.plugins", [])
    plugins = dict()
    seen = set()
    for entry_point in entry_points:
        if entry_point.name != "plugins":
            raise EntryPointsError(
                f"metagraph.plugin found an unexpected entry_point: {entry_point.name}"
            )
        elif entry_point not in seen:
            plugin_loader = entry_point.load()
            entry_point_plugins = plugin_loader()

            for entry_point_plugin_name in entry_point_plugins.keys():
                if entry_point_plugin_name in plugins:
                    raise ValueError(f"{entry_point_plugin_name} already registered.")
            plugins.update(entry_point_plugins)
            seen.add(entry_point)

    return plugins
