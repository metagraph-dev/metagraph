from metagraph import ConcreteType, Wrapper, dtypes, SequentialNodes
from metagraph.types import Matrix, Graph, WEIGHT_CHOICES
from metagraph.plugins import has_scipy
import numpy as np


if has_scipy:
    import scipy.sparse as ss

    class ScipyMatrixType(ConcreteType, abstract=Matrix):
        value_type = ss.spmatrix
        abstract_property_specificity_limits = {
            "is_dense": False,
        }

        @classmethod
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                nrows, ncols = obj.shape
                is_square = nrows == ncols
                is_symmetric = is_square and (obj.T != obj).nnz == 0
                dtype = dtypes.dtypes_simplified[obj.dtype]
                ret_val.abstract_instance = Matrix(
                    dtype=dtype, is_square=is_square, is_symmetric=is_symmetric
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

        @classmethod
        def compare_objects(cls, obj1, obj2):
            if not isinstance(obj1, cls.value_type) or not isinstance(
                obj2, cls.value_type
            ):
                raise TypeError("objects must be scipy.spmatrix")

            if obj1.dtype != obj2.dtype:
                return False
            if issubclass(obj1.dtype.type, np.floating):
                d1 = obj1.tocsr()
                d2 = obj2.tocsr()
                if not (d1.indptr == d2.indptr).all():
                    return False
                if not (d1.indices == d2.indices).all():
                    return False
                return np.isclose(d1.data, d2.data).all()
            else:
                return (obj1 != obj2).nnz == 0

    class ScipyAdjacencyMatrix(Wrapper, Graph.Mixins, abstract=Graph):
        def __init__(
            self,
            data,
            transposed=False,
            *,
            weights=None,
            is_directed=None,
            node_index=None,
        ):
            self._assert_instance(data, ss.spmatrix)
            nrows, ncols = data.shape
            self._assert(nrows == ncols, "Adjacency Matrix must be square")
            self.value = data
            self.transposed = transposed
            self._dtype = dtypes.dtypes_simplified[data.dtype]
            self._weights = self._determine_weights(weights)
            self._is_directed = self._determine_is_directed(is_directed)
            self._node_index = node_index

        def _determine_weights(self, weights):
            if weights is not None:
                if weights not in WEIGHT_CHOICES:
                    raise ValueError(f"Illegal weights: {weights}")
                return weights

            if self._dtype == "str":
                return "any"
            values = self.value.data
            if self._dtype == "bool":
                if values.all():
                    return "unweighted"
                return "non-negative"
            else:
                min_val = values.min()
                if min_val < 0:
                    return "any"
                elif min_val == 0:
                    return "non-negative"
                else:
                    if self._dtype == "int" and min_val == 1 and values.max() == 1:
                        return "unweighted"
                    return "positive"

        def _determine_is_directed(self, is_directed):
            if is_directed is not None:
                return is_directed

            return (self.value.T != self.value).nnz > 0

        @property
        def node_index(self):
            if self._node_index is None:
                self._node_index = SequentialNodes(self.value.shape[0])
            return self._node_index

        @property
        def format(self):
            return self.value.format

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
