import numpy as np
from metagraph import translator
from metagraph.plugins import has_grblas, has_scipy
from ..numpy.types import NumpyVector, NumpyNodes


if has_grblas:
    import grblas
    from .types import (
        GrblasAdjacencyMatrix,
        GrblasMatrixType,
        GrblasVectorType,
        GrblasNodes,
        dtype_mg_to_grblas,
    )

    @translator
    def vector_from_numpy(x: NumpyVector, **props) -> GrblasVectorType:
        idx = np.arange(len(x))[~x.get_missing_mask()]
        vals = x.value[idx]
        vec = grblas.Vector.new_from_values(
            idx, vals, size=len(x), dtype=dtype_mg_to_grblas[x.value.dtype]
        )
        return vec

    @translator
    def nodes_from_numpy(x: NumpyNodes, **props) -> GrblasNodes:
        idx = np.arange(len(x.value))[~x.get_missing_mask()]
        vals = x.value[idx]
        vec = grblas.Vector.new_from_values(
            idx, vals, size=len(x.value), dtype=dtype_mg_to_grblas[x.value.dtype]
        )
        return GrblasNodes(vec, weights=x._weights, node_index=x.node_index)


if has_grblas and has_scipy:
    from ..scipy.types import ScipyAdjacencyMatrix, ScipyMatrixType
    from .types import dtype_mg_to_grblas

    @translator
    def graph_from_scipy(x: ScipyAdjacencyMatrix, **props) -> GrblasAdjacencyMatrix:
        m = x.value.tocoo()
        nrows, ncols = m.shape
        dtype = dtype_mg_to_grblas[x.value.dtype]
        out = grblas.Matrix.new_from_values(
            m.row, m.col, m.data, nrows=nrows, ncols=ncols, dtype=dtype
        )
        return GrblasAdjacencyMatrix(
            out,
            transposed=x.transposed,
            weights=x._weights,
            is_directed=x._is_directed,
            node_index=x.node_index,
        )

    @translator
    def matrix_from_scipy(x: ScipyMatrixType, **props) -> GrblasMatrixType:
        x = x.tocoo()
        nrows, ncols = x.shape
        dtype = dtype_mg_to_grblas[x.dtype]
        vec = grblas.Matrix.new_from_values(
            x.row, x.col, x.data, nrows=nrows, ncols=ncols, dtype=dtype
        )
        return vec
