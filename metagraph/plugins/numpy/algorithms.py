import numpy as np
from typing import Any
from metagraph import concrete_algorithm
from .wrappers import (
    NumpyVector,
    NumpySparseVector,
    NumpyMatrix,
    NumpySparseMatrix,
)


@concrete_algorithm("casting.vector_dense_to_sparse")
def vector_dense_to_sparse(vec: NumpyVector, missing_value: Any) -> NumpySparseVector:
    return NumpySparseVector(vec.value, missing_value=missing_value)


@concrete_algorithm("casting.vector_sparse_to_dense")
def vector_sparse_to_dense(svec: NumpySparseVector, fill_value: Any) -> NumpyVector:
    return NumpyVector(np.where(svec.get_missing_mask(), fill_value, svec.value))


@concrete_algorithm("casting.matrix_dense_to_sparse")
def matrix_dense_to_sparse(m: NumpyMatrix, missing_value: Any) -> NumpySparseMatrix:
    return NumpySparseMatrix(m.value, missing_value=missing_value)


@concrete_algorithm("casting.matrix_sparse_to_dense")
def matrix_sparse_to_dense(smat: NumpySparseMatrix, fill_value: Any) -> NumpyMatrix:
    return NumpyMatrix(np.where(smat.get_missing_mask(), fill_value, smat.value))
