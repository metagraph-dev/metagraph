import numpy as np
from metagraph import translator
from metagraph.plugins import has_scipy, has_grblas
from .types import NumpyMatrix, NumpyVector, NumpyNodeMap, CompactNumpyNodeMap
from ..python.types import PythonNodeMap
from metagraph import SequentialNodes


@translator
def nodes_from_compactnodes(x: CompactNumpyNodeMap, **props) -> NumpyNodeMap:
    data = np.empty((len(x.node_index),), dtype=x.value.dtype)
    indexer = np.empty((len(x.lookup),), dtype=np.int32)
    nidx = x.node_index
    for label, pos in x.lookup.items():
        idx = nidx.bylabel(label)
        indexer[pos] = idx
    data[indexer] = x.value

    if x.num_nodes == len(x.value):
        # Dense nodes; no need for missing mask
        missing = None
    else:
        missing = np.ones_like(data, dtype=bool)
        missing[indexer] = False

    return NumpyNodeMap(
        data, missing_mask=missing, weights=x._weights, node_index=x.node_index
    )


@translator
def compactnodes_from_nodes(x: NumpyNodeMap, **props) -> CompactNumpyNodeMap:
    nidx = x.node_index
    if x.missing_mask is None:
        data = x.value
        if type(nidx) is SequentialNodes:
            lookup = {i: i for i in nidx}
        else:
            lookup = x.node_index._bylabel.copy()
    else:
        data = x.value[~x.missing_mask]
        indexes = np.arange(x.num_nodes)[~x.missing_mask]
        lookup = {nidx.byindex(idx): pos for pos, idx in enumerate(indexes)}
    return CompactNumpyNodeMap(
        data, lookup, weights=x._weights, node_index=x.node_index
    )


@translator
def compactnodes_from_python(x: PythonNodeMap, **props) -> CompactNumpyNodeMap:
    np_dtype = x._dtype if x._dtype != "str" else "object"
    data = np.empty((len(x.value),), dtype=np_dtype)
    lookup = {}
    pyvals = x.value
    for idx, label in enumerate(pyvals):
        data[idx] = pyvals[label]
        lookup[label] = idx
    return CompactNumpyNodeMap(
        data, lookup, weights=x._weights, node_index=x.node_index
    )


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
