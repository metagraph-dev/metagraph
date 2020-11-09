from metagraph import translator, dtypes
from metagraph.plugins import has_grblas
from .types import PythonNodeMapType, PythonNodeSetType, dtype_casting
from ..numpy.types import NumpyNodeMap, NumpyNodeSet
import numpy as np


@translator
def nodemap_to_nodeset(x: PythonNodeMapType, **props) -> PythonNodeSetType:
    return set(x)


@translator
def nodeset_from_numpy(x: NumpyNodeSet, **props) -> PythonNodeSetType:
    return set(x.value)


@translator
def nodemap_from_numpy(x: NumpyNodeMap, **props) -> PythonNodeMapType:
    cast = dtype_casting[dtypes.dtypes_simplified[x.value.dtype]]
    return {nid: cast(val) for nid, val in zip(x.nodes, x.value)}


@translator
def nodeset_from_numpy_nodemap(x: NumpyNodeMap, **props) -> PythonNodeSetType:
    return set(x.nodes)


if has_grblas:
    from ..graphblas.types import GrblasNodeMap

    @translator
    def nodemap_from_graphblas(x: GrblasNodeMap, **props) -> PythonNodeMapType:
        idx, vals = x.value.to_values()
        data = dict(zip(idx, vals))
        return data
