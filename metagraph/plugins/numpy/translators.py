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
        x = x.copy().astype(float)  # don't modify original
        data = x.toarray()
        # Modify x to be a 1/0 mask array
        x.data = np.ones_like(x.data)
        mask = x.toarray()
        data[mask == 0] = np.nan  # default missing value
        return NumpyMatrix(data, missing_value=np.nan)


if has_grblas:
    from ..graphblas.types import GrblasVectorType

    @translator
    def vector_from_graphblas(x: GrblasVectorType, **props) -> NumpyVector:
        inds, vals = x.to_values()
        data = np.empty((x.size,))
        data[:] = np.nan  # default missing value
        for idx, val in zip(inds, vals):
            data[idx] = val
        if len(vals) == len(data):
            # This will register as dense
            return NumpyVector(data)
        else:
            return NumpyVector(data, missing_value=np.nan)
