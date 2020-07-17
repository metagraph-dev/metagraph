import numpy as np
from metagraph import translator
from metagraph.plugins import has_scipy, has_grblas
from .types import NumpyMatrix, NumpyVector, NumpyNodeMap
from ..python.types import PythonNodeMap, PythonNodeSet


@translator
def nodemap_to_pynodeset(x: NumpyNodeMap, **props) -> PythonNodeSet:
    if x.mask is not None:
        nodes = set(np.flatnonzero(x.mask))
    elif x.id2pos is not None:
        nodes = set(x.id2pos)
    else:
        nodes = set(range(len(x.value)))
    return PythonNodeSet(nodes)


@translator
def nodemap_from_python(x: PythonNodeMap, **props) -> NumpyNodeMap:
    dtype = x._determine_dtype()
    np_dtype = dtype if dtype != "str" else "object"
    data = np.empty((len(x.value),), dtype=np_dtype)
    lookup = {}
    pyvals = x.value
    for pos, node_id in enumerate(pyvals):
        data[pos] = pyvals[node_id]
        lookup[node_id] = pos
    return NumpyNodeMap(data, node_ids=lookup)


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
        return NumpyMatrix(data, mask=existing_mask)


if has_grblas:
    from ..graphblas.types import GrblasVectorType, dtype_grblas_to_mg

    @translator
    def vector_from_graphblas(x: GrblasVectorType, **props) -> NumpyVector:
        inds, vals = x.to_values()
        data = np.empty((x.size,), dtype=dtype_grblas_to_mg[x.dtype.name])
        if len(vals) == len(data):
            for idx, val in zip(inds, vals):
                data[idx] = val
            return NumpyVector(data)
        else:
            existing_mask = np.zeros_like(data, dtype=bool)
            for idx, val in zip(inds, vals):
                data[idx] = val
                existing_mask[idx] = True
            return NumpyVector(data, mask=existing_mask)
