import warnings

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

    if _grblas.__version__ < "v1.3.2":  # pragma: no cover
        warnings.warn(
            f"grblas {_grblas.__version__} is installed, but >=v1.3.2 is required"
        )
        raise ImportError("grblas requires at least version 1.3.2")
    _grblas.init("suitesparse")

    has_grblas = True
except ImportError:  # pragma: no cover
    has_grblas = False

try:
    import numba as _

    has_numba = True
except ImportError:  # pragma: no cover
    has_numba = False

################
# Load Plugins #
################

import metagraph

# Use this as the entry_point object
registry = metagraph.PluginRegistry("core")


def find_plugins():
    from . import core, graphblas, networkx, numpy, pandas, python, scipy

    # Default Plugins
    registry.register_from_modules(core)
    registry.register_from_modules(graphblas, name="core_graphblas")
    registry.register_from_modules(networkx, name="core_networkx")
    registry.register_from_modules(numpy, name="core_numpy")
    registry.register_from_modules(pandas, name="core_pandas")
    registry.register_from_modules(python, name="core_python")
    registry.register_from_modules(scipy, name="core_scipy")

    return registry.plugins
