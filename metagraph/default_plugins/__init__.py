from ..core.plugin_registry import PluginRegistry

# Use this as the entry_point object
registry = PluginRegistry()


def find_plugins():
    # Ensure we import all items we want registered
    from . import abstract_types, wrappers, translators, algorithms

    registry.register_from_module(abstract_types)
    registry.register_from_module(wrappers)
    registry.register_from_module(translators)
    registry.register_from_module(algorithms)
    return registry.plugins


############################
# Libraries used as plugins
############################
try:
    import numpy
except ImportError:
    numpy = None

try:
    import scipy
    import scipy.sparse
except ImportError:
    scipy = None

try:
    import networkx
except ImportError:
    networkx = None

try:
    import pandas
except ImportError:
    pandas = None

try:
    import cugraph
except ImportError:
    cugraph = None

try:
    import cudf
except ImportError:
    cudf = None

try:
    import grblas

    grblas.init("suitesparse")
except ImportError:
    grblas = None
