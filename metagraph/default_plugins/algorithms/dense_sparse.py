from typing import Any
from ... import abstract_algorithm, concrete_algorithm
from ..abstract_types import DenseVector, SparseVector, DenseMatrix, SparseMatrix
from .. import registry, numpy


@abstract_algorithm("casting.vector_dense_to_sparse", registry=registry)
def vec2sparsevec(vec: DenseVector, missing_value: Any) -> SparseVector:
    pass


@abstract_algorithm("casting.vector_sparse_to_dense", registry=registry)
def sparsevec2vec(svec: SparseVector, fill_value: Any) -> DenseVector:
    pass


@abstract_algorithm("casting.matrix_dense_to_sparse", registry=registry)
def mat2sparsemat(m: DenseMatrix, missing_value: Any) -> SparseMatrix:
    pass


@abstract_algorithm("casting.matrix_sparse_to_dense", registry=registry)
def sparsemat2mat(smat: SparseMatrix, fill_value: Any) -> DenseMatrix:
    pass


######################
# Concrete Algorithms
######################

if numpy:
    np = numpy

    from ..wrappers.numpy import (
        NumpyVector,
        NumpySparseVector,
        NumpyMatrix,
        NumpySparseMatrix,
    )

    @concrete_algorithm("casting.vector_dense_to_sparse", registry=registry)
    def vec2sparsevec_np(vec: NumpyVector, missing_value: Any) -> NumpySparseVector:
        return NumpySparseVector(vec.value, missing_value=missing_value)

    @concrete_algorithm("casting.vector_sparse_to_dense", registry=registry)
    def sparsevec2vec_np(svec: NumpySparseVector, fill_value: Any) -> NumpyVector:
        return NumpyVector(np.where(svec.get_missing_mask(), fill_value, svec.value))

    @concrete_algorithm("casting.matrix_dense_to_sparse", registry=registry)
    def mat2sparsemat_np(m: NumpyMatrix, missing_value: Any) -> NumpySparseMatrix:
        return NumpySparseMatrix(m.value, missing_value=missing_value)

    @concrete_algorithm("casting.matrix_sparse_to_dense", registry=registry)
    def sparsemat2mat_np(smat: NumpySparseMatrix, fill_value: Any) -> NumpyMatrix:
        return NumpyMatrix(np.where(smat.get_missing_mask(), fill_value, smat.value))
