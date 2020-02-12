from typing import Any
from ... import PluginRegistry, abstract_algorithm, concrete_algorithm
from ..abstract_types import VectorType, SparseVectorType, MatrixType, SparseMatrixType

reg = PluginRegistry("metagraph_core")


@reg.register
@abstract_algorithm("casting.Vector_to_SparseVector")
def vec2sparsevec(vec: VectorType, missing_value: Any) -> SparseVectorType:
    pass


@reg.register
@abstract_algorithm("casting.SparseVector_to_Vector")
def sparsevec2vec(svec: SparseVectorType, missing_value: Any) -> VectorType:
    pass


@reg.register
@abstract_algorithm("casting.Matrix_to_SparseMatrix")
def mat2sparsemat(m: MatrixType, missing_value: Any) -> SparseMatrixType:
    pass


@reg.register
@abstract_algorithm("casting.SparseMatrix_to_Matrix")
def sparsemat2mat(smat: SparseMatrixType, missing_value: Any) -> MatrixType:
    pass


######################
# Concrete Algorithms
######################

try:
    import numpy as np
    from ..concrete_types.numpyobj import (
        NumpyVectorType,
        NumpySparseVectorType,
        NumpyMatrixType,
        NumpySparseMatrixType,
        NumpyVector,
        NumpySparseVector,
        NumpyMatrix,
        NumpySparseMatrix,
    )
except ImportError:
    pass
else:

    @reg.register
    @concrete_algorithm("casting.Vector_to_SparseVector")
    def vec2sparsevec_np(
        vec: NumpyVectorType, missing_value: Any
    ) -> NumpySparseVectorType:
        return NumpySparseVector(vec.obj, missing_value=missing_value)

    @reg.register
    @concrete_algorithm("casting.SparseVector_to_Vector")
    def sparsevec2vec_np(
        svec: NumpySparseVectorType, missing_value: Any
    ) -> NumpyVectorType:
        if svec.missing_value == svec.missing_value:
            x = np.where(svec.obj == svec.missing_value, missing_value, svec.obj)
        else:
            # Special handling for np.nan
            x = np.where(svec.obj != svec.obj, missing_value, svec.obj)
        return NumpyVector(x)

    @reg.register
    @concrete_algorithm("casting.Matrix_to_SparseMatrix")
    def mat2sparsemat_np(
        m: NumpyMatrixType, missing_value: Any
    ) -> NumpySparseMatrixType:
        return NumpySparseMatrix(m.obj, missing_value=missing_value)

    @reg.register
    @concrete_algorithm("casting.SparseMatrix_to_Matrix")
    def sparsemat2mat_np(
        smat: NumpySparseMatrixType, missing_value: Any
    ) -> NumpyMatrixType:
        if smat.missing_value == smat.missing_value:
            x = np.where(smat.obj == smat.missing_value, missing_value, smat.obj)
        else:
            # Special handling for np.nan
            x = np.where(smat.obj != smat.obj, missing_value, smat.obj)
        return NumpyMatrix(x)
