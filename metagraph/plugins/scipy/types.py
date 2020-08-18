from typing import Set, Dict, Any
from metagraph import ConcreteType, dtypes
from metagraph.types import Matrix, EdgeSet, EdgeMap, Graph
from metagraph.wrappers import EdgeSetWrapper, EdgeMapWrapper, CompositeGraphWrapper
from metagraph.plugins import has_scipy
import numpy as np


if has_scipy:
    import scipy.sparse as ss

    class ScipyMatrixType(ConcreteType, abstract=Matrix):
        value_type = ss.spmatrix

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: Set[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"is_dense", "dtype", "is_square"} - ret.keys():
                if prop == "is_dense":
                    ret[prop] = False
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[obj.dtype]
                if prop == "is_square":
                    nrows, ncols = obj.shape
                    ret[prop] = nrows == ncols

            # slow properties, only compute if asked
            for prop in props - ret.keys():
                if prop == "is_symmetric":
                    ret[prop] = ret["is_square"] and (obj.T != obj).nnz == 0

            return ret

        @classmethod
        def assert_equal(
            cls,
            obj1,
            obj2,
            aprops1,
            aprops2,
            cprops1,
            cprops2,
            *,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            assert obj1.shape == obj2.shape, f"{obj1.shape} != {obj2.shape}"
            assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
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
            super().__init__()
            self._assert_instance(data, ss.spmatrix)
            nrows, ncols = data.shape
            self._assert(nrows == ncols, "Adjacency Matrix must be square")
            self.value = data
            self.transposed = transposed
            if node_list is None:
                node_list = np.arange(nrows)
            elif not isinstance(node_list, np.ndarray):
                node_list = np.array(node_list)
            self.node_list = node_list

        def copy(self):
            return ScipyEdgeSet(
                self.value.copy(),
                node_list=self.node_list.copy(),
                transposed=self.transposed,
            )

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # slow properties, only compute if asked
                for prop in props - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = (obj.value.T != obj.value).nnz > 0

                return ret

            @classmethod
            def assert_equal(
                cls,
                obj1,
                obj2,
                aprops1,
                aprops2,
                cprops1,
                cprops2,
                *,
                rel_tol=None,
                abs_tol=None,
            ):
                m1, m2 = obj1.value, obj2.value
                assert (
                    m1.shape[0] == m2.shape[0]
                ), f"size mismatch: {m1.shape[0]} != {m2.shape[0]}"
                assert m1.nnz == m2.nnz, f"num edges mismatch: {m1.nnz} != {m2.nnz}"
                assert (
                    obj1.node_list == obj2.node_list
                ).all(), f"node list mismatch: {obj1.node_list} != {obj2.node_list}"
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
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
            super().__init__()
            self._assert_instance(data, ss.spmatrix)
            nrows, ncols = data.shape
            self._assert(nrows == ncols, "Adjacency Matrix must be square")
            self.value = data
            self.transposed = transposed
            if node_list is None:
                node_list = np.arange(nrows)
            elif not isinstance(node_list, np.ndarray):
                node_list = np.array(node_list)
            self._assert(
                nrows == len(node_list),
                f"node list size ({len(node_list)}) and data matrix ({nrows}) size don't match.",
            )
            self.node_list = node_list

        def copy(self):
            node_list = (
                self.node_list if self.node_list is None else self.node_list.copy()
            )
            return ScipyEdgeMap(
                self.value.copy(), node_list=node_list, transposed=self.transposed
            )

        @property
        def format(self):
            return self.value.format

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {"dtype"} - ret.keys():
                    if prop == "dtype":
                        ret[prop] = dtypes.dtypes_simplified[obj.value.dtype]

                # slow properties, only compute if asked
                for prop in props - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = (obj.value.T != obj.value).nnz > 0
                    if prop == "has_negative_weights":
                        if ret["dtype"] in {"bool", "str"}:
                            neg_weights = None
                        else:
                            min_val = obj.value.data.min()
                            if min_val < 0:
                                neg_weights = True
                            else:
                                neg_weights = False
                        ret[prop] = neg_weights

                return ret

            @classmethod
            def assert_equal(
                cls,
                obj1,
                obj2,
                aprops1,
                aprops2,
                cprops1,
                cprops2,
                *,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                m1, m2 = obj1.value, obj2.value
                assert (
                    m1.shape[0] == m2.shape[0]
                ), f"size mismatch: {m1.shape[0]} != {m2.shape[0]}"
                assert m1.nnz == m2.nnz, f"num edges mismatch: {m1.nnz} != {m2.nnz}"
                assert (
                    obj1.node_list == obj2.node_list
                ).all(), f"node list mismatch: {obj1.node_list} != {obj2.node_list}"
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
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
                    assert np.isclose(
                        d1.data, d2.data, rtol=rel_tol, atol=abs_tol
                    ).all()
                else:
                    assert (d1.data == d2.data).all()

    class ScipyGraph(CompositeGraphWrapper, abstract=Graph):
        def __init__(self, edges, nodes=None):
            # Import here to avoid circular import
            from ..numpy.types import NumpyNodeSet, NumpyNodeMap

            # Auto convert simple matrix to EdgeMap
            # Anything more complicated requires explicit creation of the EdgeMap or EdgeSet
            if isinstance(edges, ss.spmatrix):
                edges = ScipyEdgeMap(edges)

            super().__init__(edges, nodes)
            self._assert_instance(edges, (ScipyEdgeSet, ScipyEdgeMap))
            if nodes is not None:
                self._assert_instance(nodes, (NumpyNodeSet, NumpyNodeMap))

        def copy(self):
            nodes = self.nodes if self.nodes is None else self.nodes.copy()
            return ScipyGraph(self.edges.copy(), nodes=nodes)
