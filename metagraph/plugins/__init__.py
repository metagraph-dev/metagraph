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

    # Default Plugins
    registry.register_from_modules(
        "graphblas_plugin", [graphblas, metagraph.types, metagraph.algorithms]
    )
    registry.register_from_modules("networkx_plugin", [networkx])
    registry.register_from_modules("numpy_plugin", [numpy])
    registry.register_from_modules("pandas_plugin", [pandas])
    registry.register_from_modules("python_plugin", [python])
    registry.register_from_modules("scipy_plugin", [scipy])

    return registry
