import metagraph
from ..core.plugin_registry import PluginRegistry

# Use this as the entry_point object
registry = PluginRegistry()


def find_plugins():
    # Ensure we import all items we want registered
    registry.register_from_modules(
        metagraph.types, metagraph.algorithms, metagraph.plugins
    )
    return registry.plugins


############################
# Libraries used as plugins
############################

try:
    import scipy.sparse as _

    has_scipy = True
except ImportError:
    has_scipy = False

try:
    import networkx as _

    has_networkx = True
except ImportError:
    has_networkx = False

try:
    import pandas as _

    has_pandas = True
except ImportError:
    has_pandas = False

try:
    import grblas as _grblas

    _grblas.init("suitesparse")
    has_grblas = True
except ImportError:
    has_grblas = False

from . import graphblas, networkx, numpy, pandas, python, scipy
