from metagraph import ConcreteType, dtypes
from metagraph.types import Vector, Matrix, NodeSet, NodeMap, EdgeSet, EdgeMap, Graph
from metagraph.wrappers import (
    NodeSetWrapper,
    NodeMapWrapper,
    EdgeSetWrapper,
    EdgeMapWrapper,
    GraphWrapper,
)
from metagraph.plugins import has_grblas

from typing import Set, Dict, Any


if has_grblas:
    import grblas

    dtype_mg_to_grblas = {
        dtypes.bool: grblas.dtypes.BOOL,
        dtypes.int8: grblas.dtypes.INT8,
        dtypes.int16: grblas.dtypes.INT16,
        dtypes.int32: grblas.dtypes.INT32,
        dtypes.int64: grblas.dtypes.INT64,
        dtypes.uint8: grblas.dtypes.UINT8,
        dtypes.uint16: grblas.dtypes.UINT16,
        dtypes.uint32: grblas.dtypes.UINT32,
        dtypes.uint64: grblas.dtypes.UINT64,
        dtypes.float32: grblas.dtypes.FP32,
        dtypes.float64: grblas.dtypes.FP64,
    }

    dtype_grblas_to_mg = {v.name: k for k, v in dtype_mg_to_grblas.items()}

    class GrblasVectorType(ConcreteType, abstract=Vector):
        value_type = grblas.Vector

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: Set[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"dtype"} - ret.keys():
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[
                        dtype_grblas_to_mg[obj.dtype.name]
                    ]

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
            if obj1.dtype.name in {"FP32", "FP64"}:
                assert obj1.isclose(
                    obj2, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=True
                )
            else:
                assert obj1.isequal(obj2, check_dtype=True)

    class GrblasNodeSet(NodeSetWrapper, abstract=NodeSet):
        def __init__(self, data, *, aprops=None):
            super().__init__(aprops=aprops)
            self._assert_instance(data, grblas.Vector)
            self.value = data

        def __len__(self):
            return self.value.nvals

        def __contains__(self, key):
            return 0 <= key < len(self.value) and self.value[key].value is not None

        class TypeMixin:
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
                v1, v2 = obj1.value, obj2.value
                assert (
                    v1.nvals == v2.nvals
                ), f"num nodes mismatch: {v1.nvals} != {v2.nvals}"
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                # Compare
                shape_match = obj1.value.ewise_mult(
                    obj2.value, grblas.binary.pair
                ).new()
                assert shape_match.nvals == v1.nvals, f"node ids do not match"

    class GrblasNodeMap(NodeMapWrapper, abstract=NodeMap):
        def __init__(self, data, *, aprops=None):
            super().__init__(aprops=aprops)
            self._assert_instance(data, grblas.Vector)
            self.value = data

        def __getitem__(self, node_id):
            return self.value[node_id].value

        def __len__(self):
            return self.value.nvals

        def __contains__(self, key):
            return 0 <= key < len(self.value) and self.value[key].value is not None

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {"dtype"} - ret.keys():
                    if prop == "dtype":
                        ret[prop] = dtypes.dtypes_simplified[
                            dtype_grblas_to_mg[obj.value.dtype.name]
                        ]

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
                v1, v2 = obj1.value, obj2.value
                assert v1.size == v2.size, f"size mismatch: {v1.size} != {v2.size}"
                assert (
                    v1.nvals == v2.nvals
                ), f"num nodes mismatch: {v1.nvals} != {v2.nvals}"
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                # Compare
                if v1.dtype.name in {"FP32", "FP64"}:
                    assert obj1.value.isclose(
                        obj2.value, rel_tol=rel_tol, abs_tol=abs_tol
                    )
                else:
                    assert obj1.value.isequal(obj2.value)

    class GrblasMatrixType(ConcreteType, abstract=Matrix):
        value_type = grblas.Matrix

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: Set[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"dtype"} - ret.keys():
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[
                        dtype_grblas_to_mg[obj.dtype.name]
                    ]

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
            if obj1.dtype.name in {"FP32", "FP64"}:
                assert obj1.isclose(
                    obj2, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=True
                )
            else:
                assert obj1.isequal(obj2, check_dtype=True)

    def find_active_nodes(m):
        """
        Given a grblas.Matrix, returns a list of the active nodes.
        Active nodes are defined as having an edge.
        """
        v = m.reduce_rows(grblas.monoid.any).new()
        h = m.reduce_columns(grblas.monoid.any).new()
        v << v.ewise_add(h, grblas.monoid.any)
        idx, _ = v.to_values()
        # TODO: revisit this once grblas returns numpy arrays directly
        return list(idx)

    class GrblasEdgeSet(EdgeSetWrapper, abstract=EdgeSet):
        """
        Matrix id is the NodeId. Only information about the edges is preserved, meaning nrows and ncols
        are irrelevant. They must be large enough to hold all nodes attached to edges, but otherwise provide
        no information about nodes which have no edges.
        The actual values in the Matrix are not used.
        """

        def __init__(self, data, *, aprops=None):
            super().__init__(aprops=aprops)
            self._assert_instance(data, grblas.Matrix)
            self._assert(data.nrows == data.ncols, "adjacency matrix must be square")
            self.value = data

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # slow properties, only compute if asked
                for prop in props - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = not obj.value.isequal(obj.value.T.new())

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
                v1, v2 = obj1.value, obj2.value
                assert (
                    v1.nvals == v2.nvals
                ), f"num nodes mismatch: {v1.nvals} != {v2.nvals}"
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                # Compare edges
                if v1.nrows != v2.nrows:
                    # Size is allowed to be different if extra nodes have no edges; trim to same size for comparison
                    if v1.nrows < v2.nrows:
                        v2 = v2.dup()
                        v2.resize(v1.nrows, v1.ncols)
                    else:
                        v1 = v1.dup()
                        v1.resize(v2.nrows, v2.ncols)
                shape_match = v1.ewise_mult(v2, grblas.binary.pair).new()
                assert shape_match.nvals == v1.nvals, "edges do not match"

    class GrblasEdgeMap(EdgeMapWrapper, abstract=EdgeMap):
        """
        Matrix id is the NodeId. Only information about the edges is preserved, meaning nrows and ncols
        are irrelevant. They must be large enough to hold all nodes attached to edges, but otherwise provide
        no information about nodes which have no edges.
        """

        def __init__(self, data, aprops=None):
            super().__init__(aprops=aprops)
            self._assert_instance(data, grblas.Matrix)
            self._assert(data.nrows == data.ncols, "adjacency matrix must be square")
            self.value = data

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {"dtype"} - ret.keys():
                    if prop == "dtype":
                        ret[prop] = dtypes.dtypes_simplified[
                            dtype_grblas_to_mg[obj.value.dtype.name]
                        ]

                # slow properties, only compute if asked
                for prop in props - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = not obj.value.isequal(obj.value.T.new())
                    if prop == "has_negative_weights":
                        if ret["dtype"] in {"bool", "str"}:
                            neg_weights = None
                        else:
                            min_val = (
                                obj.value.reduce_scalar(grblas.monoid.min).new().value
                            )
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
                v1, v2 = obj1.value, obj2.value
                assert (
                    v1.nvals == v2.nvals
                ), f"num nodes mismatch: {v1.nvals} != {v2.nvals}"
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                # Compare edges
                if v1.nrows != v2.nrows:
                    # Size is allowed to be different if extra nodes have no edges; trim to same size for comparison
                    if v1.nrows < v2.nrows:
                        v2 = v2.dup()
                        v2.resize(v1.nrows, v1.ncols)
                    else:
                        v1 = v1.dup()
                        v1.resize(v2.nrows, v2.ncols)
                if v1.dtype.name in {"FP32", "FP64"}:
                    assert v1.isclose(v2, rel_tol=rel_tol, abs_tol=abs_tol)
                else:
                    assert v1.isequal(v2)

    class GrblasGraph(GraphWrapper, abstract=Graph):
        """
        Matrix positional index is the NodeId. Only information about the edges is preserved in the Matrix.
        Information about which nodes are present in the graph are contained in the nodes Vector. NodeIds
        missing from `nodes` are assumed to be missing in the graph. Otherwise they are contained in the graph,
        including isolate nodes.
        """

        def __init__(self, matrix, nodes=None, *, aprops=None):
            """
            matrix: grblas.Matrix
            nodes: grblas.Vector

            nodes represent the active nodes and possibly their values. If not set, all nodes in the matrix
                are considered as part of the graph. If nodes is provided, all non-missing items are considered
                as active nodes in the graph. In other words, the structural mask is used, not the values.
            """
            super().__init__(aprops=aprops)
            self._assert_instance(matrix, grblas.Matrix)
            nrows, ncols = matrix.nrows, matrix.ncols
            self._assert(
                nrows == ncols, f"Adjacency matrix must be square, not {nrows}x{ncols}"
            )
            if nodes is None:
                nodes = grblas.Vector.from_values(
                    range(nrows), [True] * nrows, size=nrows, dtype=bool
                )
            self._assert_instance(nodes, grblas.Vector)
            self._assert(
                nodes.size == matrix.nrows,
                f"nodes length {nodes.size} must match matrix size {nrows}",
            )
            self.value = matrix
            self.nodes = nodes

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
                        ret[prop] = "set" if obj.nodes.dtype == bool else "map"
                    elif prop == "edge_type":
                        ret[prop] = "set" if obj.value.dtype == bool else "map"

                # Delegate to GrblasEdge{Set/Map} to compute edge properties
                if ret["edge_type"] == "set":
                    ret["edge_dtype"] = None
                    ret["edge_has_negative_weights"] = None
                    edgeclass = GrblasEdgeSet
                else:
                    edgeclass = GrblasEdgeMap
                edge_props = {
                    cls._edge_prop_map[p] for p in props if p in cls._edge_prop_map
                }
                known_edge_props = {
                    cls._edge_prop_map[p]: v
                    for p, v in known_props.items()
                    if p in cls._edge_prop_map
                }
                edges = edgeclass(obj.value)
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
                            ret[prop] = dtypes.dtypes_simplified[
                                dtype_grblas_to_mg[obj.nodes.dtype.name]
                            ]

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
                    edgeset1 = GrblasEdgeSet(obj1.value)
                    edgeset2 = GrblasEdgeSet(obj2.value)
                    GrblasEdgeSet.Type.assert_equal(
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
                    edgemap1 = GrblasEdgeMap(obj1.value)
                    edgemap2 = GrblasEdgeMap(obj2.value)
                    GrblasEdgeMap.Type.assert_equal(
                        edgemap1,
                        edgemap2,
                        subprops1,
                        subprops2,
                        {},
                        {},
                        rel_tol=rel_tol,
                        abs_tol=abs_tol,
                    )

                # Compare active nodes
                nodes1 = obj1.nodes
                nodes2 = obj2.nodes
                assert (
                    nodes1.nvals == nodes2.nvals
                ), f"num active nodes mismatch: {nodes1.nvals} != {nodes2.nvals}"
                # Ensure same size nodes
                if nodes1.size != nodes2.size:
                    # Size is allowed to be different if extra nodes have no edges; trim to same size for comparison
                    if nodes1.size < nodes2.size:
                        nodes2 = nodes2.dup()
                        nodes2.resize(nodes1.size)
                    else:
                        nodes1 = nodes1.dup()
                        nodes1.resize(nodes2.size)

                assert (
                    nodes1.nvals == nodes2.nvals
                ), f"num active nodes mismatch after resize {nodes1.nvals} != {nodes2.nvals}"
                if aprops1["node_type"] == "map":
                    # Compare
                    if nodes1.dtype.name in {"FP32", "FP64"}:
                        assert nodes1.isclose(nodes2, rel_tol=rel_tol, abs_tol=abs_tol)
                    else:
                        assert nodes1.isequal(nodes2)
                elif aprops1["node_type"] == "set":
                    shape_match = nodes1.ewise_mult(nodes2, grblas.binary.pair).new()
                    assert (
                        shape_match.nvals == nodes1.nvals
                    ), "active nodes do not match"
