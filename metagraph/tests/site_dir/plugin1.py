import metagraph

# Use this as the entry_point object
registry = metagraph.PluginRegistry("plugin1_default_plugin_name")


def find_plugins():
    import plugin1_util

    registry.register_from_modules(
        plugin1_util, metagraph.types, metagraph.algorithms, name="plugin1"
    )
    return registry.plugins
