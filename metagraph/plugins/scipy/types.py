from typing import List, Dict, Any
from metagraph import ConcreteType, dtypes
from metagraph.types import Matrix, EdgeSet, EdgeMap
from metagraph.wrappers import EdgeSetWrapper, EdgeMapWrapper
from metagraph.plugins import has_scipy
import numpy as np


if has_scipy:
    import scipy.sparse as ss

    class ScipyMatrixType(ConcreteType, abstract=Matrix):
        value_type = ss.spmatrix
        abstract_property_specificity_limits = {"is_dense": False}

        @classmethod
        def compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            cls._validate_abstract_props(props)
            nrows, ncols = obj.shape
            is_square = nrows == ncols
            dtype = dtypes.dtypes_simplified[obj.dtype]

            ret = dict(dtype=dtype, is_square=is_square)

            # slow properties, only compute if asked
            for propname in props:
                if propname == "is_symmetric":
                    ret["is_symmetric"] = is_square and (obj.T != obj).nnz == 0

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0):
            assert obj1.shape == obj2.shape, f"{obj1.shape} != {obj2.shape}"
            if issubclass(obj1.dtype.type, np.floating):
                d1 = obj1.tocsr()
                d2 = obj2.tocsr()
                # Check shape
                assert (d1.indptr == d2.indptr).all(), f"{d1.indptr == d2.indptr}"
                assert (d1.indices == d2.indices).all(), f"{d1.indices == d2.indices}"
                assert np.isclose(d1.data, d2.data, rtol=rel_tol, atol=abs_tol).all()
            else:
                # Recommended way to check for equality
                assert (obj1 != obj2).nnz == 0, f"{(obj1 != obj2).toarray()}"

    class ScipyEdgeSet(EdgeSetWrapper, abstract=EdgeSet):
        def __init__(self, data, node_list=None, transposed=False):
            self._assert_instance(data, ss.spmatrix)
            nrows, ncols = data.shape
            self._assert(nrows == ncols, "Adjacency Matrix must be square")
            self.value = data
            self.transposed = transposed
            if node_list is None:
                node_list = np.arange(nrows)
            elif not isinstance(node_list, np.ndarray):
                node_list = np.array(node_list)
            self._node_list = node_list

        @classmethod
        def assert_equal(cls, obj1, obj2, *, rel_tol=None, abs_tol=None):
            m1, m2 = obj1.value, obj2.value
            assert (
                m1.shape[0] == m2.shape[0]
            ), f"size mismatch: {m1.shape[0]} != {m2.shape[0]}"
            assert m1.nnz == m2.nnz, f"num edges mismatch: {m1.nnz} != {m2.nnz}"
            assert (
                obj1._node_list == obj2._node_list
            ).all(), f"node list mismatch: {obj1._node_list} != {obj2._node_list}"
            # Handle transposed states
            d1 = m1.T if obj1.transposed else m1
            d2 = m2.T if obj2.transposed else m2
            # Compare
            d1 = d1.tocsr()
            d2 = d2.tocsr()
            assert (d1.indptr == d2.indptr).all(), f"{d1.indptr == d2.indptr}"
            # Ensure sorted indices for numpy matching to work
            d1.sort_indices()
            d2.sort_indices()
            assert (d1.indices == d2.indices).all(), f"{d1.indices == d2.indices}"

    class ScipyEdgeMap(EdgeMapWrapper, abstract=EdgeMap):
        def __init__(
            self, data, node_list=None, transposed=False,
        ):
            self._assert_instance(data, ss.spmatrix)
            nrows, ncols = data.shape
            self._assert(nrows == ncols, "Adjacency Matrix must be square")
            self.value = data
            self.transposed = transposed
            if node_list is None:
                node_list = np.arange(nrows)
            elif not isinstance(node_list, np.ndarray):
                node_list = np.array(node_list)
            self._node_list = node_list

        def to_edgeset(self):
            return ScipyEdgeSet(self.value, self._node_list, self.transposed)

        @property
        def format(self):
            return self.value.format

        @classmethod
        def compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            cls._validate_abstract_props(props)

            # fast properties
            ret = {"dtype": dtypes.dtypes_simplified[obj.value.dtype]}

            # slow properties, only compute if asked
            if "is_directed" in props:
                ret["is_directed"] = (obj.value.T != obj.value).nnz > 0
            if "weights" in props:
                if ret["dtype"] == "str":
                    weights = "any"
                values = obj.value.data
                if ret["dtype"] == "bool":
                    weights = "non-negative"
                else:
                    min_val = values.min()
                    if min_val < 0:
                        weights = "any"
                    elif min_val == 0:
                        weights = "non-negative"
                    else:
                        weights = "positive"
                ret["weights"] = weights

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0):
            m1, m2 = obj1.value, obj2.value
            assert (
                m1.shape[0] == m2.shape[0]
            ), f"size mismatch: {m1.shape[0]} != {m2.shape[0]}"
            assert m1.nnz == m2.nnz, f"num edges mismatch: {m1.nnz} != {m2.nnz}"
            assert (
                obj1._node_list == obj2._node_list
            ).all(), f"node list mismatch: {obj1._node_list} != {obj2._node_list}"
            # Handle transposed states
            d1 = m1.T if obj1.transposed else m1
            d2 = m2.T if obj2.transposed else m2
            # Compare
            d1 = d1.tocsr()
            d2 = d2.tocsr()
            assert (d1.indptr == d2.indptr).all(), f"{d1.indptr == d2.indptr}"
            # Ensure sorted indices for numpy matching to work
            d1.sort_indices()
            d2.sort_indices()
            assert (d1.indices == d2.indices).all(), f"{d1.indices == d2.indices}"
            if issubclass(d1.dtype.type, np.floating):
                assert np.isclose(d1.data, d2.data, rtol=rel_tol, atol=abs_tol).all()
            else:
                assert (d1.data == d2.data).all()
