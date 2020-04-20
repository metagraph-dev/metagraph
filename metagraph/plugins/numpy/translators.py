import numpy as np
from metagraph import translator
from metagraph.plugins import has_scipy, has_grblas
from .types import NumpyMatrix, NumpyVector, NumpyNodes
from ..python.types import PythonNodes


@translator
def nodes_from_python(x: PythonNodes, **props) -> NumpyNodes:
    np_dtype = x._dtype if x._dtype != "str" else "object"
    data = np.empty((len(x.value),), dtype=np_dtype)
    for idx, label in enumerate(x.node_index):
        data[idx] = x.value[label]
    return NumpyNodes(data, weights=x._weights, node_index=x.node_index)


if has_scipy:
    from ..scipy.types import ScipyMatrixType

    @translator
    def matrix_from_scipy(x: ScipyMatrixType, **props) -> NumpyMatrix:
        # This is trickier than simply calling .toarray() because
        # scipy.sparse assumes empty means zero
        # Mask is required To properly handle any non-empty zeros
        existing = x.copy().astype(bool)  # don't modify original
        data = x.toarray()
        existing.data = np.ones_like(existing.data)
        existing_mask = existing.toarray()
        return NumpyMatrix(data, missing_mask=~existing_mask)


if has_grblas:
    from ..graphblas.types import GrblasVectorType, dtype_grblas_to_mg

    @translator
    def vector_from_graphblas(x: GrblasVectorType, **props) -> NumpyVector:
        inds, vals = x.to_values()
        data = np.empty((x.size,), dtype=dtype_grblas_to_mg[x.dtype])
        if len(vals) == len(data):
            for idx, val in zip(inds, vals):
                data[idx] = val
            return NumpyVector(data)
        else:
            missing_mask = np.ones_like(data, dtype=bool)
            for idx, val in zip(inds, vals):
                data[idx] = val
                missing_mask[idx] = False
            return NumpyVector(data, missing_mask=missing_mask)
