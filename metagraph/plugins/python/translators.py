from metagraph import translator
from metagraph.plugins import has_grblas
from .types import PythonNodeMap, dtype_casting
from ..numpy.types import CompactNumpyNodeMap


@translator
def nodes_from_compactnumpy(x: CompactNumpyNodeMap, **props) -> PythonNodeMap:
    cast = dtype_casting[x._dtype]
    npdata = x.value
    nplookup = x.lookup
    data = {label: cast(npdata[idx]) for label, idx in nplookup.items()}
    return PythonNodeMap(
        data, dtype=x._dtype, weights=x._weights, node_index=x.node_index
    )


if has_grblas:
    from ..graphblas.types import GrblasNodeMap

    @translator
    def nodes_from_graphblas(x: GrblasNodeMap, **props) -> PythonNodeMap:
        idx, vals = x.value.to_values()
        idx2label = x.node_index.byindex
        data = {idx2label(k): v for k, v in zip(idx, vals)}
        return PythonNodeMap(
            data, dtype=x._dtype, weights=x._weights, node_index=x.node_index
        )
