from metagraph import translator, dtypes
from metagraph.plugins import has_grblas
from .types import PythonNodeMap, PythonNodeSet, dtype_casting
from ..numpy.types import NumpyNodeMap


@translator
def nodemap_to_nodeset(x: PythonNodeMap, **props) -> PythonNodeSet:
    return PythonNodeSet(set(x.value))


@translator
def nodemap_from_compactnumpy(x: NumpyNodeMap, **props) -> PythonNodeMap:
    cast = dtype_casting[dtypes.dtypes_simplified[x.value.dtype]]
    npdata = x.value
    nplookup = x.id2pos
    data = {label: cast(npdata[idx]) for label, idx in nplookup.items()}
    return PythonNodeMap(data)


if has_grblas:
    from ..graphblas.types import GrblasNodeMap

    @translator
    def nodemap_from_graphblas(x: GrblasNodeMap, **props) -> PythonNodeMap:
        idx, vals = x.value.to_values()
        data = dict(zip(idx, vals))
        return PythonNodeMap(data)
