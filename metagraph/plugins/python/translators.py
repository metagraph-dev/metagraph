from metagraph import translator
from metagraph.plugins import has_grblas
from .types import PythonNodes, dtype_casting
from ..numpy.types import NumpyNodes


@translator
def nodes_from_numpy(x: NumpyNodes, **props) -> PythonNodes:
    cast = dtype_casting[x._dtype]
    idx2label = x.node_index.byindex
    data = {
        idx2label(idx): cast(x.value[idx])
        for idx, is_missing in enumerate(x.get_missing_mask())
        if not is_missing
    }
    return PythonNodes(
        data, dtype=x._dtype, weights=x._weights, node_index=x.node_index
    )


if has_grblas:
    from ..graphblas.types import GrblasNodes

    @translator
    def nodes_from_graphblas(x: GrblasNodes, **props) -> PythonNodes:
        idx, vals = x.value.to_values()
        idx2label = x.node_index.byindex
        data = {idx2label(k): v for k, v in zip(idx, vals)}
        return PythonNodes(
            data, dtype=x._dtype, weights=x._weights, node_index=x.node_index
        )
