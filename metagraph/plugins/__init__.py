############################
# Libraries used as plugins
############################

try:
    import scipy.sparse as _

    has_scipy = True
except ImportError:  # pragma: no cover
    has_scipy = False

try:
    import networkx as _

    has_networkx = True
except ImportError:  # pragma: no cover
    has_networkx = False

try:
    import community as _

    has_community = True
except ImportError:  # pragma: no cover
    has_community = False

try:
    import pandas as _

    has_pandas = True
except ImportError:  # pragma: no cover
    has_pandas = False

try:
    import grblas as _grblas

    _grblas.init("suitesparse")
    has_grblas = True
except ImportError:  # pragma: no cover
    has_grblas = False

################
# Load Plugins #
################

import metagraph

# Use this as the entry_point object
registry = metagraph.PluginRegistry()


def find_plugins():
    from . import graphblas, networkx, numpy, pandas, python, scipy

    # TODO create & use PluginRegistry.register_abstract_from_modules to handle strictly
    # abstract types and algorithms since metagraph.types and metagraph.algorithms
    # only contain those and the plugin_name arg isn't used
    # also rename PluginRegistry.register_from_modules -> PluginRegistry.register_concrete_from_modules
    registry.register_from_modules(
        "seeing_this_plugin_name_indicates_bug", [metagraph.types, metagraph.algorithms]
    )

    # Default Plugins
    registry.register_from_modules("graphblas_plugin", [graphblas])
    registry.register_from_modules("networkx_plugin", [networkx])
    registry.register_from_modules("numpy_plugin", [numpy])
    registry.register_from_modules("pandas_plugin", [pandas])
    registry.register_from_modules("python_plugin", [python])
    registry.register_from_modules("scipy_plugin", [scipy])

    return registry
