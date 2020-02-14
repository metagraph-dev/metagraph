from typing import Any
from ... import abstract_algorithm, concrete_algorithm
from ..abstract_types import DenseVector, SparseVector, DenseMatrix, SparseMatrix
from .. import registry


@abstract_algorithm("casting.vector_dense_to_sparse", registry=registry)
def vec2sparsevec(vec: DenseVector, missing_value: Any) -> SparseVector:
    pass


@abstract_algorithm("casting.vector_sparse_to_dense", registry=registry)
def sparsevec2vec(svec: SparseVector, missing_value: Any) -> DenseVector:
    pass


@abstract_algorithm("casting.matrix_dense_to_sparse", registry=registry)
def mat2sparsemat(m: DenseMatrix, missing_value: Any) -> SparseMatrix:
    pass


@abstract_algorithm("casting.matrix_sparse_to_dense", registry=registry)
def sparsemat2mat(smat: SparseMatrix, missing_value: Any) -> DenseMatrix:
    pass


######################
# Concrete Algorithms
######################

try:
    import numpy as np
    from ..wrappers.numpy import (
        NumpyVector,
        NumpySparseVector,
        NumpyMatrix,
        NumpySparseMatrix,
    )
except ImportError:
    pass
else:

    @concrete_algorithm("casting.vector_dense_to_sparse", registry=registry)
    def vec2sparsevec_np(vec: NumpyVector, missing_value: Any) -> NumpySparseVector:
        return NumpySparseVector(vec.value, missing_value=missing_value)

    @concrete_algorithm("casting.vector_sparse_to_dense", registry=registry)
    def sparsevec2vec_np(svec: NumpySparseVector, missing_value: Any) -> NumpyVector:
        if svec.missing_value == svec.missing_value:
            x = np.where(svec.value == svec.missing_value, missing_value, svec.value)
        else:
            # Special handling for np.nan
            x = np.where(svec.value != svec.value, missing_value, svec.value)
        return NumpyVector(x)

    @concrete_algorithm("casting.matrix_dense_to_sparse", registry=registry)
    def mat2sparsemat_np(m: NumpyMatrix, missing_value: Any) -> NumpySparseMatrix:
        return NumpySparseMatrix(m.value, missing_value=missing_value)

    @concrete_algorithm("casting.matrix_sparse_to_dense", registry=registry)
    def sparsemat2mat_np(smat: NumpySparseMatrix, missing_value: Any) -> NumpyMatrix:
        if smat.missing_value == smat.missing_value:
            x = np.where(smat.value == smat.missing_value, missing_value, smat.value)
        else:
            # Special handling for np.nan
            x = np.where(smat.value != smat.value, missing_value, smat.value)
        return NumpyMatrix(x)
