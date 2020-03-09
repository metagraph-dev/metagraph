import numpy as np
from metagraph import translator
from metagraph.plugins import has_scipy, has_grblas
from .wrappers import NumpySparseMatrix, NumpySparseVector, dtype_np_to_mg
from ..python.wrappers import PythonSparseVector


@translator
def sparsevector_from_python(x: PythonSparseVector, **props) -> NumpySparseVector:
    data = np.empty((len(x),))
    data[:] = np.nan  # default missing value
    for idx, val in x.value.items():
        data[idx] = val
    return NumpySparseVector(data, missing_value=np.nan)


if has_scipy:
    import scipy.sparse as ss
    from ..scipy.wrappers import ScipySparseMatrixType

    @translator
    def sparsematrix_from_scipy(x: ScipySparseMatrixType, **props) -> NumpySparseMatrix:
        # This is trickier than simply calling .toarray() because
        # scipy.sparse assumes empty means zero
        # Mask is required To properly handle any non-empty zeros
        x = x.copy().astype(float)  # don't modify original
        data = x.toarray()
        # Modify x to be a 1/0 mask array
        x.data = np.ones_like(x.data)
        mask = x.toarray()
        data[mask == 0] = np.nan  # default missing value
        return NumpySparseMatrix(data, missing_value=np.nan)


if has_grblas:
    from ..graphblas.wrappers import GrblasVectorType

    @translator
    def sparsevector_from_graphblas(x: GrblasVectorType, **props) -> NumpySparseVector:
        inds, vals = x.to_values()
        data = np.empty((x.size,))
        data[:] = np.nan  # default missing value
        for idx, val in zip(inds, vals):
            data[idx] = val
        return NumpySparseVector(data, missing_value=np.nan)
