from typing import Any
from metagraph import abstract_algorithm
from metagraph.types import DenseVector, SparseVector, DenseMatrix, SparseMatrix


@abstract_algorithm("casting.vector_dense_to_sparse")
def vec2sparsevec(vec: DenseVector, missing_value: Any) -> SparseVector:
    pass


@abstract_algorithm("casting.vector_sparse_to_dense")
def sparsevec2vec(svec: SparseVector, fill_value: Any) -> DenseVector:
    pass


@abstract_algorithm("casting.matrix_dense_to_sparse")
def mat2sparsemat(m: DenseMatrix, missing_value: Any) -> SparseMatrix:
    pass


@abstract_algorithm("casting.matrix_sparse_to_dense")
def sparsemat2mat(smat: SparseMatrix, fill_value: Any) -> DenseMatrix:
    pass
