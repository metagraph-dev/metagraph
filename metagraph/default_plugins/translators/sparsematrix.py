from ... import translator
from ..wrappers.scipy import ScipySparseMatrixType
from ..wrappers.numpy import NumpySparseMatrix, dtype_np_to_mg
from ..wrappers.graphblas import GrblasMatrixType, dtype_mg_to_grblas
from .. import numpy, scipy, grblas


if scipy and numpy:
    ss = scipy.sparse
    np = numpy

    @translator
    def translate_sparsematrix_sci2np(
        x: ScipySparseMatrixType, **props
    ) -> NumpySparseMatrix:
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

    @translator
    def translate_sparsematrix_np2sci(
        x: NumpySparseMatrix, **props
    ) -> ScipySparseMatrixType:
        # scipy.sparse assumes zero mean empty
        # To work around this limitation, we use a mask
        # and directly set .data after construction
        non_mask = ~x.get_missing_mask()
        mat = ss.coo_matrix(non_mask)
        mat.data = x.value[non_mask]
        return mat


if scipy and grblas:
    ss = scipy.sparse

    @translator
    def translate_sparsematrix_sci2grb(
        x: ScipySparseMatrixType, **props
    ) -> GrblasMatrixType:
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

    @translator
    def translate_sparsematrix_grb2sci(
        x: GrblasMatrixType, **props
    ) -> ScipySparseMatrixType:
        rows, cols, vals = x.to_values()
        mat = ss.coo_matrix((tuple(vals), (tuple(rows), tuple(cols))), x.shape)
        return mat
