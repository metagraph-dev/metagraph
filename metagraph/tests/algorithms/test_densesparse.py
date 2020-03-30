import numpy as np


# TODO: update these once methods are available: fillna, sparsify

# def test_vector_dense_to_sparse(default_plugin_resolver):
#     dpr = default_plugin_resolver
#     x = dpr.wrapper.DenseVector.NumpyVector(np.array([0, 4, 2, 3, 4, 99]))
#     y = dpr.algo.casting.vector_dense_to_sparse(x, missing_value=4)
#     assert y.nnz == 4
#
#
# def test_vector_sparse_to_dense(default_plugin_resolver):
#     dpr = default_plugin_resolver
#     x = dpr.wrapper.SparseVector.NumpySparseVector(
#         np.array([0, 4, 2, 3, 4, 99]), missing_value=4
#     )
#     y = dpr.algo.casting.vector_sparse_to_dense(x, fill_value=42)
#     assert (y.value == np.array([0, 42, 2, 3, 42, 99])).all()
#
#
# def test_matrix_dense_to_sparse(default_plugin_resolver):
#     dpr = default_plugin_resolver
#     m = np.array([[1, 2, 3], [2, 5, 6], [7, 8, 2]])
#     x = dpr.wrapper.DenseMatrix.NumpyMatrix(m)
#     y = dpr.algo.casting.matrix_dense_to_sparse(x, missing_value=2)
#     assert y.nnz == 6
#
#
# def test_matrix_sparse_to_dense(default_plugin_resolver):
#     dpr = default_plugin_resolver
#     m = np.array([[1, 2, 3, 4], [2, 5, 6, -4], [7, 8, 2, 2]])
#     x = dpr.wrapper.SparseMatrix.NumpySparseMatrix(m, missing_value=2)
#     y = dpr.algo.casting.matrix_sparse_to_dense(x, fill_value=0)
#     result = np.array([[1, 0, 3, 4], [0, 5, 6, -4], [7, 8, 0, 0]])
#     assert (y.value == result).all().all()
