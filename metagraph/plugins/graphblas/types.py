from metagraph import ConcreteType, Wrapper, dtypes, SequentialNodes
from metagraph.types import Vector, Nodes, NodeMapping, Matrix, Graph, WEIGHT_CHOICES
from metagraph.plugins import has_grblas


if has_grblas:
    import grblas

    dtype_grblas_to_mg = {
        grblas.dtypes.BOOL: dtypes.bool,
        grblas.dtypes.INT8: dtypes.int8,
        grblas.dtypes.INT16: dtypes.int16,
        grblas.dtypes.INT32: dtypes.int32,
        grblas.dtypes.INT64: dtypes.int64,
        grblas.dtypes.UINT8: dtypes.uint8,
        grblas.dtypes.UINT16: dtypes.uint16,
        grblas.dtypes.UINT32: dtypes.uint32,
        grblas.dtypes.UINT64: dtypes.uint64,
        grblas.dtypes.FP32: dtypes.float32,
        grblas.dtypes.FP64: dtypes.float64,
    }
    dtype_mg_to_grblas = {v: k for k, v in dtype_grblas_to_mg.items()}

    class GrblasVectorType(ConcreteType, abstract=Vector):
        value_type = grblas.Vector

        @classmethod
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                is_dense = obj.nvals == obj.size
                dtype = dtypes.dtypes_simplified[dtype_grblas_to_mg[obj.dtype]]
                ret_val.abstract_instance = cls.abstract(is_dense=is_dense, dtype=dtype)
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

        @classmethod
        def compare_objects(cls, obj1, obj2):
            if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
                raise TypeError("objects must be grblas.Vector")

            if obj1.dtype in {"FP32", "FP64"}:
                return obj1.isclose(obj2, check_dtype=True)
            else:
                return obj1.isequal(obj2, check_dtype=True)

    class GrblasNodes(Wrapper, Nodes.Mixins, abstract=Nodes):
        def __init__(self, data, *, weights=None, node_index=None):
            self._assert_instance(data, grblas.Vector)
            self.value = data
            self._dtype = dtypes.dtypes_simplified[dtype_grblas_to_mg[data.dtype]]
            self._weights = self._determine_weights(weights)
            self._node_index = node_index

        def __getitem__(self, label):
            if self._node_index is None:
                return self.value[label].value
            return self.value[self._node_index.bylabel(label)].value

        def _determine_weights(self, weights=None):
            if weights is not None:
                if weights not in WEIGHT_CHOICES:
                    raise ValueError(f"Illegal weights: {weights}")
                return weights

            if self._dtype == "str":
                return "any"
            if self._dtype == "bool":
                if self.value.reduce(grblas.monoid.land).new():
                    return "unweighted"
                return "non-negative"
            else:
                min_val = self.value.reduce(grblas.monoid.min).new().value
                if min_val < 0:
                    return "any"
                elif min_val == 0:
                    return "non-negative"
                else:
                    max_val = self.value.reduce(grblas.monoid.max).new().value
                    if self._dtype == "int" and min_val == 1 and max_val == 1:
                        return "unweighted"
                    return "positive"

        @property
        def num_nodes(self):
            return self.value.size

        @property
        def node_index(self):
            if self._node_index is None:
                self._node_index = SequentialNodes(self.num_nodes)
            return self._node_index

        def rebuild_for_node_index(self, node_index):
            """
            Returns a new instance based on `node_index`
            """
            if self.num_nodes != len(node_index):
                raise ValueError(
                    f"Size of node_index ({len(node_index)}) must match num_nodes ({self.num_nodes})"
                )

            data = self.value
            if node_index != self.node_index:
                my_node_index = self.node_index
                my_node_index._verify_valid_conversion(node_index)
                index_converter = [my_node_index.bylabel(label) for label in node_index]
                data = data[index_converter].new()
            return GrblasNodes(data, weights=self._weights, node_index=node_index)

        @classmethod
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                ret_val.abstract_instance = cls.abstract(
                    dtype=obj._dtype, weights=obj._weights
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

        @classmethod
        def compare_objects(cls, obj1, obj2):
            if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
                raise TypeError("objects must be GrblasNodes")

            if obj1.num_nodes != obj2.num_nodes:
                return False
            if obj1._dtype != obj2._dtype or obj1._weights != obj2._weights:
                return False
            # Convert to a common node indexing scheme
            try:
                obj2 = obj2.rebuild_for_node_index(obj1.node_index)
            except ValueError:
                return False
            # Compare
            if obj1._dtype == "float":
                return obj1.value.isclose(obj2.value)
            else:
                return obj1.value.isequal(obj2.value)

    class GrblasNodeMapping(Wrapper, abstract=NodeMapping):
        def __init__(self, data, src_node_labels=None, dst_node_labels=None):
            self.value = data
            self.src_node_labels = src_node_labels
            self.dst_node_labels = dst_node_labels

    class GrblasMatrixType(ConcreteType, abstract=Matrix):
        value_type = grblas.Matrix
        abstract_property_specificity_limits = {
            "is_dense": False,
        }

        @classmethod
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                is_square = obj.nrows == obj.ncols
                is_symmetric = obj == obj.T.new()
                dtype = dtypes.dtypes_simplified[dtype_grblas_to_mg[obj.dtype]]
                ret_val.abstract_instance = Matrix(
                    dtype=dtype, is_square=is_square, is_symmetric=is_symmetric
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

        @classmethod
        def compare_objects(cls, obj1, obj2):
            if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
                raise TypeError("objects must be grblas.Matrix")

            if obj1.dtype in {"FP32", "FP64"}:
                return obj1.isclose(obj2, check_dtype=True)
            else:
                return obj1.isequal(obj2, check_dtype=True)

    class GrblasAdjacencyMatrix(Wrapper, Graph.Mixins, abstract=Graph):
        def __init__(
            self,
            data,
            transposed=False,
            *,
            weights=None,
            is_directed=None,
            node_index=None,
        ):
            self._assert_instance(data, grblas.Matrix)
            self._assert(data.nrows == data.ncols, "Adjacency Matrix must be square")
            self.value = data
            self.transposed = transposed
            self._node_index = node_index
            self._dtype = dtypes.dtypes_simplified[dtype_grblas_to_mg[data.dtype]]
            self._weights = self._determine_weights(weights)
            self._is_directed = self._determine_is_directed(is_directed)

        def _determine_weights(self, weights):
            if weights is not None:
                if weights not in WEIGHT_CHOICES:
                    raise ValueError(f"Illegal weights: {weights}")
                return weights

            if self._dtype == "str":
                return "any"
            if self._dtype == "bool":
                if self.value.reduce_scalar(grblas.monoid.land).new():
                    return "unweighted"
                return "non-negative"
            else:
                min_val = self.value.reduce_scalar(grblas.monoid.min).new().value
                if min_val < 0:
                    return "any"
                elif min_val == 0:
                    return "non-negative"
                else:
                    max_val = self.value.reduce_scalar(grblas.monoid.max).new().value
                    if self._dtype == "int" and min_val == 1 and max_val == 1:
                        return "unweighted"
                    return "positive"

        def _determine_is_directed(self, is_directed):
            if is_directed is not None:
                return is_directed

            return self.value != self.value.T.new()

        def dtype(self):
            return dtype_grblas_to_mg[self.value.dtype]

        def show(self):
            return self.value.show()

        @property
        def node_index(self):
            if self._node_index is None:
                self._node_index = SequentialNodes(self.value.nrows)
            return self._node_index

        @classmethod
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                ret_val.abstract_instance = Graph(
                    dtype=obj._dtype, weights=obj._weights, is_directed=obj._is_directed
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")
