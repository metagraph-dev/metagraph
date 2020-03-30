from metagraph import ConcreteType, Wrapper, dtypes
from metagraph.types import Matrix, Graph, WEIGHT_CHOICES
from metagraph.plugins import has_scipy


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
                is_symmetric = (obj.T != obj).nnz == 0
                dtype = dtypes.dtypes_simplified[obj.value.dtype]
                ret_val.abstract_instance = Matrix(
                    dtype=dtype, is_square=is_square, is_symmetric=is_symmetric
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

    class ScipyAdjacencyMatrix(Wrapper, abstract=Graph):
        def __init__(self, data, transposed=False, *, weights=None, is_directed=None):
            self.value = data
            self.transposed = transposed
            self._assert_instance(data, ss.spmatrix)
            self._dtype = dtypes.dtypes_simplified[data.dtype]
            self._weights = self._determine_weights(weights)
            self._is_directed = self._determine_is_directed(is_directed)

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
