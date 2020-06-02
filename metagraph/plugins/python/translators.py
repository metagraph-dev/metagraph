from metagraph import translator, dtypes
from metagraph.plugins import has_grblas
from .types import PythonNodeMap, dtype_casting
from ..numpy.types import CompactNumpyNodeMap


@translator
def nodemap_from_compactnumpy(x: CompactNumpyNodeMap, **props) -> PythonNodeMap:
    cast = dtype_casting[dtypes.dtypes_simplified[x.value.dtype]]
    npdata = x.value
    nplookup = x.lookup
    data = {label: cast(npdata[idx]) for label, idx in nplookup.items()}
    return PythonNodeMap(data)


if has_grblas:
    from ..graphblas.types import GrblasNodeMap

    @translator
    def nodemap_from_graphblas(x: GrblasNodeMap, **props) -> PythonNodeMap:
        idx, vals = x.value.to_values()
        data = dict(zip(idx, vals))
        return PythonNodeMap(data)
