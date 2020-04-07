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

    class GrblasNodes(Wrapper, abstract=Nodes):
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
        def node_index(self):
            if self._node_index is None:
                self._node_index = SequentialNodes(self.value.size)
            return self._node_index

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

    class GrblasAdjacencyMatrix(Wrapper, abstract=Graph):
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
