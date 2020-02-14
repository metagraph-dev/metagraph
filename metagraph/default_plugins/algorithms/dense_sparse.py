from typing import Any
from ... import abstract_algorithm, concrete_algorithm
from ..abstract_types import DenseVector, SparseVector, DenseMatrix, SparseMatrix
from .. import registry


@abstract_algorithm("casting.Vector_Dense_to_Sparse", registry=registry)
def vec2sparsevec(vec: DenseVector, missing_value: Any) -> SparseVector:
    pass


@abstract_algorithm("casting.Vector_Sparse_to_Dense", registry=registry)
def sparsevec2vec(svec: SparseVector, missing_value: Any) -> DenseVector:
    pass


@abstract_algorithm("casting.Matrix_Dense_to_Sparse", registry=registry)
def mat2sparsemat(m: DenseMatrix, missing_value: Any) -> SparseMatrix:
    pass


@abstract_algorithm("casting.Matrix_Sparse_to_Dense", registry=registry)
def sparsemat2mat(smat: SparseMatrix, missing_value: Any) -> DenseMatrix:
    pass


######################
# Concrete Algorithms
######################

try:
    import numpy as np
    from ..wrappers.numpyobj import (
        NumpyVector,
        NumpySparseVector,
        NumpyMatrix,
        NumpySparseMatrix,
    )
except ImportError:
    pass
else:

    @concrete_algorithm("casting.Vector_Dense_to_Sparse")
    def vec2sparsevec_np(vec: NumpyVector, missing_value: Any) -> NumpySparseVector:
        return NumpySparseVector(vec.obj, missing_value=missing_value)

    @concrete_algorithm("casting.Vector_Sparse_to_Dense")
    def sparsevec2vec_np(svec: NumpySparseVector, missing_value: Any) -> NumpyVector:
        if svec.missing_value == svec.missing_value:
            x = np.where(svec.obj == svec.missing_value, missing_value, svec.obj)
        else:
            # Special handling for np.nan
            x = np.where(svec.obj != svec.obj, missing_value, svec.obj)
        return NumpyVector(x)

    @concrete_algorithm("casting.Matrix_Dense_to_Sparse")
    def mat2sparsemat_np(m: NumpyMatrix, missing_value: Any) -> NumpySparseMatrix:
        return NumpySparseMatrix(m.obj, missing_value=missing_value)

    @concrete_algorithm("casting.Matrix_Sparse_to_Dense")
    def sparsemat2mat_np(smat: NumpySparseMatrix, missing_value: Any) -> NumpyMatrix:
        if smat.missing_value == smat.missing_value:
            x = np.where(smat.obj == smat.missing_value, missing_value, smat.obj)
        else:
            # Special handling for np.nan
            x = np.where(smat.obj != smat.obj, missing_value, smat.obj)
        return NumpyMatrix(x)
