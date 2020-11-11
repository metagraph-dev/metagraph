import numpy as np
from metagraph import translator
from metagraph.plugins import has_scipy, has_grblas
from .types import NumpyMatrixType, NumpyVectorType, NumpyNodeSet, NumpyNodeMap
from ..python.types import PythonNodeMapType, PythonNodeSetType


@translator
def nodemap_to_nodeset(x: NumpyNodeMap, **props) -> NumpyNodeSet:
    return NumpyNodeSet(x.nodes.copy())


@translator
def nodeset_from_python(x: PythonNodeSetType, **props) -> NumpyNodeSet:
    return NumpyNodeSet(x)


@translator
def nodemap_from_python(x: PythonNodeMapType, **props) -> NumpyNodeMap:
    aprops = PythonNodeMapType.compute_abstract_properties(x, {"dtype"})
    dtype = aprops["dtype"]
    np_dtype = dtype if dtype != "str" else "object"
    data = np.empty((len(x),), dtype=np_dtype)
    nodes = np.empty((len(x),), dtype=int)
    for i, (node_id, val) in enumerate(x.items()):
        nodes[i] = node_id
        data[i] = val
    return NumpyNodeMap(data, nodes=nodes, aprops=aprops)


if has_grblas:
    from ..graphblas.types import (
        GrblasVectorType,
        GrblasNodeSet,
        GrblasNodeMap,
        GrblasMatrixType,
        dtype_grblas_to_mg,
    )

    @translator
    def vector_from_graphblas(x: GrblasVectorType, **props) -> NumpyVectorType:
        _, vals = x.to_values()
        return np.array(vals)

    @translator
    def nodeset_from_graphblas(x: GrblasNodeSet, **props) -> NumpyNodeSet:
        idx, _ = x.value.to_values()
        return NumpyNodeSet(idx)

    @translator
    def nodemap_from_graphblas(x: GrblasNodeMap, **props) -> NumpyNodeMap:
        idx, vals = x.value.to_values()
        # TODO: remove this line once `to_values()` returns ndarray
        vals = np.array(vals, dtype=dtype_grblas_to_mg[x.value.dtype.name])
        return NumpyNodeMap(vals, nodes=idx)

    @translator
    def matrix_from_grblas(x: GrblasMatrixType, **props) -> NumpyMatrixType:
        _, _, vals = x.to_values()
        # TODO: adjust this once `to_values()` returns ndarray
        vals = np.array(vals, dtype=dtype_grblas_to_mg[x.dtype.name]).reshape(
            (x.nrows, x.ncols)
        )
        return vals
