from typing import Set, Dict, Any
from metagraph import ConcreteType, dtypes
from metagraph.types import Matrix, EdgeSet, EdgeMap, Graph
from metagraph.wrappers import EdgeSetWrapper, EdgeMapWrapper, GraphWrapper
from metagraph.plugins import has_scipy
import numpy as np


if has_scipy:
    import scipy.sparse as ss

    class ScipyEdgeSet(EdgeSetWrapper, abstract=EdgeSet):
        """
        scipy.sparse matrix is the minimal size to contain all edges.
        If nodes are not sequential, a node_list must be provided to map the matrix index to NodeId.
        Nodes which are present in the matrix but have no edges are not allowed as
            they will not survive a roundtrip translation.
        The actual values in the matrix are unused.
        """

        def __init__(self, data, node_list=None, *, aprops=None):
            super().__init__(aprops=aprops)
            self._assert_instance(data, ss.spmatrix)
            nrows, ncols = data.shape
            self._assert(nrows == ncols, "Adjacency Matrix must be square")
            self.value = data
            if node_list is None:
                node_list = np.arange(nrows)
            else:
                self._assert_instance(node_list, (np.ndarray, list, tuple))
                if not isinstance(node_list, np.ndarray):
                    node_list = np.array(node_list)
            self._assert(
                nrows == len(node_list),
                f"node list size ({len(node_list)}) and data matrix size ({nrows}) don't match.",
            )
            self.node_list = node_list

        def copy(self):
            return ScipyEdgeSet(self.value.copy(), node_list=self.node_list.copy())

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
                if not (obj1.node_list == obj2.node_list).all():
                    nl1 = set(obj1.node_list)
                    nl2 = set(obj2.node_list)
                    assert nl1 == nl2, f"node list mismatch: {nl1 ^ nl2}"
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                # Convert to COO format, apply node ids, then sort to allow comparison of indices
                d1 = m1.tocoo()
                d2 = m2.tocoo()
                r1, c1 = obj1.node_list[d1.row], obj1.node_list[d1.col]
                r2, c2 = obj2.node_list[d2.row], obj2.node_list[d2.col]
                sort1 = np.lexsort((c1, r1))
                sort2 = np.lexsort((c2, r2))
                r1, c1 = r1[sort1], c1[sort1]
                r2, c2 = r2[sort2], c2[sort2]
                assert (r1 == r2).all(), f"rows mismatch {r1} != {r2}"
                assert (c1 == c2).all(), f"cols mismatch {c1} != {c2}"

    class ScipyEdgeMap(EdgeMapWrapper, abstract=EdgeMap):
        """
        scipy.sparse matrix is the minimal size to contain all edges.
        If nodes are not sequential, a node_list must be provided to map the matrix index to NodeId.
        Nodes which are present in the matrix but have no edges are not allowed as
            they will not survive a roundtrip translation.
        """

        def __init__(self, data, node_list=None, *, aprops=None):
            super().__init__(aprops=aprops)
            self._assert_instance(data, ss.spmatrix)
            nrows, ncols = data.shape
            self._assert(nrows == ncols, "Adjacency Matrix must be square")
            self.value = data
            if node_list is None:
                node_list = np.arange(nrows)
            else:
                self._assert_instance(node_list, (np.ndarray, list, tuple))
                if not isinstance(node_list, np.ndarray):
                    node_list = np.array(node_list)
            self._assert(
                nrows == len(node_list),
                f"node list size ({len(node_list)}) and data matrix size ({nrows}) don't match.",
            )
            self.node_list = node_list

        def copy(self):
            node_list = (
                self.node_list if self.node_list is None else self.node_list.copy()
            )
            return ScipyEdgeMap(self.value.copy(), node_list=node_list)

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
                if not (obj1.node_list == obj2.node_list).all():
                    nl1 = set(obj1.node_list)
                    nl2 = set(obj2.node_list)
                    assert nl1 == nl2, f"node list mismatch: {nl1 ^ nl2}"
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                # Convert to COO format, apply node ids, then sort to allow comparison of indices and values
                d1 = m1.tocoo()
                d2 = m2.tocoo()
                r1, c1, v1 = obj1.node_list[d1.row], obj1.node_list[d1.col], d1.data
                r2, c2, v2 = obj2.node_list[d2.row], obj2.node_list[d2.col], d2.data
                sort1 = np.lexsort((c1, r1))
                sort2 = np.lexsort((c2, r2))
                r1, c1, v1 = r1[sort1], c1[sort1], v1[sort1]
                r2, c2, v2 = r2[sort2], c2[sort2], v2[sort2]
                assert (r1 == r2).all(), f"rows mismatch {r1} != {r2}"
                assert (c1 == c2).all(), f"cols mismatch {c1} != {c2}"
                if issubclass(d1.dtype.type, np.floating):
                    assert np.isclose(v1, v2, rtol=rel_tol, atol=abs_tol).all()
                else:
                    assert (v1 == v2).all()

    class ScipyGraph(GraphWrapper, abstract=Graph):
        """
        scipy.sparse matrix is the minimal size to contain all nodes in the graph.
        If nodes are not sequential, a node_list must be provided to map the matrix index to NodeId.
        node_vals (if populated) contains node weights
        """

        def __init__(self, matrix, node_list=None, node_vals=None, *, aprops=None):
            super().__init__(aprops=aprops)
            self._assert_instance(matrix, ss.spmatrix)
            nrows, ncols = matrix.shape
            self._assert(
                nrows == ncols, f"adjacency matrix must be square, not {nrows}x{ncols}"
            )
            if node_list is None:
                node_list = np.arange(nrows)
            else:
                self._assert_instance(node_list, (np.ndarray, list, tuple))
                if not isinstance(node_list, np.ndarray):
                    node_list = np.array(node_list)
            self._assert(
                nrows == len(node_list),
                f"node list size ({len(node_list)}) and data matrix size ({nrows}) don't match.",
            )
            if node_vals is not None:
                self._assert_instance(node_vals, (np.ndarray, list, tuple))
                if not isinstance(node_vals, np.ndarray):
                    node_vals = np.array(node_vals)
                self._assert(
                    nrows == len(node_vals),
                    f"node vals size ({len(node_vals)}) and data matrix size ({nrows}) don't match",
                )
            self.value = matrix
            self.node_list = node_list
            self.node_vals = node_vals

        def copy(self):
            node_vals = None if self.node_vals is None else self.node_vals.copy()
            return ScipyGraph(self.value.copy(), self.node_list.copy(), node_vals)

        class TypeMixin:
            # Both forward and reverse lookup
            _edge_prop_map = {
                "is_directed": "is_directed",
                "edge_dtype": "dtype",
                "edge_has_negative_weights": "has_negative_weights",
                "dtype": "edge_dtype",
                "has_negative_weights": "edge_has_negative_weights",
            }

            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {"node_type", "edge_type"} - ret.keys():
                    if prop == "node_type":
                        ret[prop] = "set" if obj.node_vals is None else "map"
                    elif prop == "edge_type":
                        ret[prop] = "set" if obj.value.dtype == bool else "map"

                # Delegate to ScipyEdge{Set/Map} to compute edge properties
                if ret["edge_type"] == "set":
                    ret["edge_dtype"] = None
                    ret["edge_has_negative_weights"] = None
                    edgeclass = ScipyEdgeSet
                else:
                    edgeclass = ScipyEdgeMap
                edge_props = {
                    cls._edge_prop_map[p] for p in props if p in cls._edge_prop_map
                }
                known_edge_props = {
                    cls._edge_prop_map[p]: v
                    for p, v in known_props.items()
                    if p in cls._edge_prop_map
                }
                edges = edgeclass(obj.value, obj.node_list)
                edge_computed_props = edgeclass.Type._compute_abstract_properties(
                    edges, edge_props, known_edge_props
                )
                ret.update(
                    {cls._edge_prop_map[p]: v for p, v in edge_computed_props.items()}
                )

                # slow properties, only compute if asked
                for prop in props - ret.keys():
                    if prop == "node_dtype":
                        if ret["node_type"] == "set":
                            ret[prop] = None
                        else:
                            ret[prop] = dtypes.dtypes_simplified[obj.node_vals.dtype]

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
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                subprops1 = {
                    cls._edge_prop_map[p]: v
                    for p, v in aprops1.items()
                    if p in cls._edge_prop_map
                }
                subprops2 = {
                    cls._edge_prop_map[p]: v
                    for p, v in aprops2.items()
                    if p in cls._edge_prop_map
                }
                if aprops1["edge_type"] == "set":
                    edgeset1 = ScipyEdgeSet(obj1.value, obj1.node_list)
                    edgeset2 = ScipyEdgeSet(obj2.value, obj2.node_list)
                    ScipyEdgeSet.Type.assert_equal(
                        edgeset1,
                        edgeset2,
                        subprops1,
                        subprops2,
                        {},
                        {},
                        rel_tol=rel_tol,
                        abs_tol=abs_tol,
                    )
                else:
                    edgemap1 = ScipyEdgeMap(obj1.value, obj1.node_list)
                    edgemap2 = ScipyEdgeMap(obj2.value, obj2.node_list)
                    ScipyEdgeMap.Type.assert_equal(
                        edgemap1,
                        edgemap2,
                        subprops1,
                        subprops2,
                        {},
                        {},
                        rel_tol=rel_tol,
                        abs_tol=abs_tol,
                    )
                if aprops1["node_type"] == "map":
                    sort1 = np.argsort(obj1.node_list)
                    sort2 = np.argsort(obj2.node_list)
                    vals1 = obj1.node_vals[sort1]
                    vals2 = obj2.node_vals[sort2]
                    if issubclass(vals1.dtype.type, np.floating):
                        assert np.isclose(
                            vals1, vals2, rtol=rel_tol, atol=abs_tol
                        ).all()
                    else:
                        assert (vals1 == vals2).all()
