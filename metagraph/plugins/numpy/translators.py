import numpy as np
from metagraph import translator
from metagraph.plugins import has_scipy, has_grblas
from .types import NumpyMatrix, NumpyVector, NumpyNodeSet, NumpyNodeMap
from ..python.types import PythonNodeMap, PythonNodeSet


@translator
def nodemap_to_nodeset(x: NumpyNodeMap, **props) -> NumpyNodeSet:
    return NumpyNodeSet(x.nodes(copy=True))


@translator
def nodeset_from_python(x: PythonNodeSet, **props) -> NumpyNodeSet:
    return NumpyNodeSet(np.array(sorted(x.value)))


@translator
def nodemap_from_python(x: PythonNodeMap, **props) -> NumpyNodeMap:
    dtype = x._determine_dtype()
    np_dtype = dtype if dtype != "str" else "object"
    data = np.empty((len(x.value),), dtype=np_dtype)
    lookup = {}
    pyvals = x.value
    for pos, node_id in enumerate(sorted(pyvals)):
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
    from ..graphblas.types import (
        GrblasVectorType,
        GrblasNodeSet,
        GrblasNodeMap,
        dtype_grblas_to_mg,
    )

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

    @translator
    def nodeset_from_graphblas(x: GrblasNodeSet, **props) -> NumpyNodeSet:
        idx, _ = x.value.to_values()
        return NumpyNodeSet(np.array(idx))

    @translator
    def nodemap_from_graphblas(x: GrblasNodeMap, **props) -> NumpyNodeMap:
        idx, vals = x.value.to_values()
        vals = np.array(vals, dtype=dtype_grblas_to_mg[x.value.dtype.name])
        return NumpyNodeMap(vals, node_ids=np.array(idx))
