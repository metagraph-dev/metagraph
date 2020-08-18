from metagraph import translator, dtypes
from metagraph.plugins import has_grblas
from .types import PythonNodeMap, PythonNodeSet, dtype_casting
from ..numpy.types import NumpyNodeMap, NumpyNodeSet
import numpy as np


@translator
def nodemap_to_nodeset(x: PythonNodeMap, **props) -> PythonNodeSet:
    return PythonNodeSet(set(x.value))


@translator
def nodeset_from_numpy(x: NumpyNodeSet, **props) -> PythonNodeSet:
    if x.mask is None:
        return PythonNodeSet(x.node_set)
    else:
        return PythonNodeSet(set(np.flatnonzero(x.mask)))


@translator
def nodemap_from_numpy(x: NumpyNodeMap, **props) -> PythonNodeMap:
    cast = dtype_casting[dtypes.dtypes_simplified[x.value.dtype]]
    npdata = x.value
    if x.mask is not None:
        nplookup = np.flatnonzero(x.mask)
        data = {idx: cast(npdata[idx]) for idx in nplookup}
    elif x.id2pos is not None:
        nplookup = x.id2pos
        data = {label: cast(npdata[idx]) for label, idx in nplookup.items()}
    else:
        data = {label: cast(npdata_elem) for label, npdata_elem in enumerate(npdata)}
    return PythonNodeMap(data)


@translator
def nodeset_from_numpy_nodemap(x: NumpyNodeMap, **props) -> PythonNodeSet:
    if x.mask is not None:
        nodes = set(np.flatnonzero(x.mask))
    elif x.id2pos is not None:
        nodes = set(x.id2pos)
    else:
        nodes = set(range(len(x.value)))
    return PythonNodeSet(nodes)


if has_grblas:
    from ..graphblas.types import GrblasNodeMap

    @translator
    def nodemap_from_graphblas(x: GrblasNodeMap, **props) -> PythonNodeMap:
        idx, vals = x.value.to_values()
        data = dict(zip(idx, vals))
        return PythonNodeMap(data)
