import numpy as np
from metagraph import translator
from metagraph.plugins import has_grblas, has_scipy
from ..numpy.types import NumpyVector, NumpyNodeMap


if has_grblas:
    import grblas
    from .types import (
        GrblasEdgeMap,
        GrblasEdgeSet,
        GrblasMatrixType,
        GrblasVectorType,
        GrblasNodeSet,
        GrblasNodeMap,
        dtype_mg_to_grblas,
    )

    @translator
    def nodemap_to_nodeset(x: GrblasNodeMap, **props) -> GrblasNodeSet:
        return GrblasNodeSet(x.value)

    @translator
    def edgemap_to_edgeset(x: GrblasEdgeMap, **props) -> GrblasEdgeSet:
        return GrblasEdgeSet(x.value, transposed=x.transposed)

    @translator
    def vector_from_numpy(x: NumpyVector, **props) -> GrblasVectorType:
        idx = np.arange(len(x))
        if x.missing_mask is not None:
            idx = idx[~x.missing_mask]
        vals = x.value[idx]
        vec = grblas.Vector.from_values(
            idx, vals, size=len(x), dtype=dtype_mg_to_grblas[x.value.dtype]
        )
        return vec

    @translator
    def nodemap_from_numpy(x: NumpyNodeMap, **props) -> GrblasNodeMap:
        idx = np.arange(len(x.value))
        if x.missing_mask is not None:
            idx = idx[~x.missing_mask]
        vals = x.value[idx]
        vec = grblas.Vector.from_values(
            idx, vals, size=len(x.value), dtype=dtype_mg_to_grblas[x.value.dtype]
        )
        return GrblasNodeMap(vec)


if has_grblas and has_scipy:
    from ..scipy.types import ScipyEdgeMap, ScipyMatrixType
    from .types import dtype_mg_to_grblas

    @translator
    def edgemap_from_scipy(x: ScipyEdgeMap, **props) -> GrblasEdgeMap:
        m = x.value.tocoo()
        node_list = x._node_list
        size = max(node_list) + 1
        dtype = dtype_mg_to_grblas[x.value.dtype]
        out = grblas.Matrix.from_values(
            node_list[m.row],
            node_list[m.col],
            m.data,
            nrows=size,
            ncols=size,
            dtype=dtype,
        )
        return GrblasEdgeMap(out, transposed=x.transposed,)

    @translator
    def matrix_from_scipy(x: ScipyMatrixType, **props) -> GrblasMatrixType:
        x = x.tocoo()
        nrows, ncols = x.shape
        dtype = dtype_mg_to_grblas[x.dtype]
        vec = grblas.Matrix.from_values(
            x.row, x.col, x.data, nrows=nrows, ncols=ncols, dtype=dtype
        )
        return vec
