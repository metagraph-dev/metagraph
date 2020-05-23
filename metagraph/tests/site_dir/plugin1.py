import metagraph

# Use this as the entry_point object
registry = metagraph.PluginRegistry()


def find_plugins():
    import plugin1_util

    registry.register_from_modules(
        "plugin1", [plugin1_util, metagraph.types, metagraph.algorithms]
    )
    return registry
