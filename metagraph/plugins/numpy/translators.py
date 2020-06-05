import numpy as np
from metagraph import translator
from metagraph.plugins import has_scipy, has_grblas
from .types import NumpyMatrix, NumpyVector, NumpyNodeMap, CompactNumpyNodeMap
from ..python.types import PythonNodeMap, PythonNodeSet


@translator
def nodemap_to_pynodeset(x: NumpyNodeMap, **props) -> PythonNodeSet:
    values = x.value
    if x.missing_mask is not None:
        values = values[~x.missing_mask]
    return PythonNodeSet(set(values))


@translator
def compactnodemap_to_pynodeset(x: CompactNumpyNodeMap, **props) -> PythonNodeSet:
    return PythonNodeSet(set(x.lookup))


@translator
def nodemap_from_compactnodemap(x: CompactNumpyNodeMap, **props) -> NumpyNodeMap:
    size = max(x.lookup) + 1
    data = np.empty((size,), dtype=x.value.dtype)
    indexer = np.empty((len(x.lookup),), dtype=np.int32)
    for node_id, pos in x.lookup.items():
        indexer[pos] = node_id
    data[indexer] = x.value

    if size == len(x.value):
        # Dense nodes; no need for missing mask
        missing = None
    else:
        missing = np.ones_like(data, dtype=bool)
        missing[indexer] = False

    return NumpyNodeMap(data, missing_mask=missing)


@translator
def compactnodemap_from_nodemap(x: NumpyNodeMap, **props) -> CompactNumpyNodeMap:
    if x.missing_mask is None:
        data = x.value
        lookup = {i: i for i in range(len(data))}
    else:
        data = x.value[~x.missing_mask]
        indexes = np.arange(len(x.value))[~x.missing_mask]
        lookup = {idx: pos for pos, idx in enumerate(indexes)}
    return CompactNumpyNodeMap(data, lookup)


@translator
def compactnodes_from_python(x: PythonNodeMap, **props) -> CompactNumpyNodeMap:
    dtype = x._determine_dtype()
    np_dtype = dtype if dtype != "str" else "object"
    data = np.empty((len(x.value),), dtype=np_dtype)
    lookup = {}
    pyvals = x.value
    for pos, node_id in enumerate(pyvals):
        data[pos] = pyvals[node_id]
        lookup[node_id] = pos
    return CompactNumpyNodeMap(data, lookup)


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
        data = np.empty((x.size,), dtype=dtype_grblas_to_mg[x.dtype.name])
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
