import numpy as np
from metagraph import translator
from metagraph.plugins import has_grblas
from .types import PythonSparseVector
from ..numpy.types import NumpySparseVector


@translator
def sparsevector_from_numpy(x: NumpySparseVector, **props) -> PythonSparseVector:
    data = {
        idx: x.value[idx]
        for idx, is_missing in enumerate(x.get_missing_mask())
        if not is_missing
    }
    return PythonSparseVector(data, size=len(x))


if has_grblas:
    from ..graphblas.types import GrblasVectorType, dtype_mg_to_grblas

    @translator
    def sparsevector_from_graphblas(x: GrblasVectorType, **props) -> PythonSparseVector:
        idx, vals = x.to_values()
        data = {k: v for k, v in zip(idx, vals)}
        return PythonSparseVector(data, size=x.size)
