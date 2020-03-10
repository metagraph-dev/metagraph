from metagraph import translator
from metagraph.plugins import has_grblas, has_scipy

if has_grblas:
    import grblas
    from .types import (
        GrblasAdjacencyMatrix,
        GrblasMatrixType,
        GrblasVectorType,
        dtype_mg_to_grblas,
    )
    from ..numpy.types import NumpySparseVector
    from ..python.types import PythonSparseVector

    @translator
    def sparsevector_from_python(x: PythonSparseVector, **props) -> GrblasVectorType:
        idx, vals = zip(*x.value.items())
        vec = grblas.Vector.new_from_values(
            idx, vals, size=len(x), dtype=dtype_mg_to_grblas[x.dtype]
        )
        return vec

    @translator
    def sparsevector_from_numpy(x: NumpySparseVector, **props) -> GrblasVectorType:
        idx = [
            idx for idx, is_missing in enumerate(x.get_missing_mask()) if not is_missing
        ]
        vals = [x.value[i] for i in idx]
        vec = grblas.Vector.new_from_values(
            idx, vals, size=len(x), dtype=dtype_mg_to_grblas[x.dtype]
        )
        return vec


if has_grblas and has_scipy:
    import scipy.sparse as ss
    from ..scipy.types import ScipyAdjacencyMatrix, ScipySparseMatrixType
    from ..numpy.types import dtype_np_to_mg

    @translator
    def graph_from_scipy(x: ScipyAdjacencyMatrix, **props) -> GrblasAdjacencyMatrix:
        m = x.value.tocoo()
        nrows, ncols = m.shape
        out = grblas.Matrix.new_from_values(
            m.row, m.col, m.data, nrows=nrows, ncols=ncols, dtype=grblas.dtypes.INT64
        )
        return GrblasAdjacencyMatrix(out, transposed=x.transposed)

    @translator
    def sparsematrix_from_scipy(x: ScipySparseMatrixType, **props) -> GrblasMatrixType:
        x = x.tocoo()
        nrows, ncols = x.shape
        dtype = dtype_np_to_mg(x.dtype)
        vec = grblas.Matrix.new_from_values(
            x.row,
            x.col,
            x.data,
            nrows=nrows,
            ncols=ncols,
            dtype=dtype_mg_to_grblas[dtype],
        )
        return vec
