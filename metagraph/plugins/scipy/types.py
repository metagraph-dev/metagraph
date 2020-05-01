from typing import List, Dict, Any
from metagraph import ConcreteType, Wrapper, dtypes, SequentialNodes
from metagraph.types import Matrix, Graph, WEIGHT_CHOICES
from metagraph.plugins import has_scipy
import numpy as np


if has_scipy:
    import scipy.sparse as ss

    class ScipyMatrixType(ConcreteType, abstract=Matrix):
        value_type = ss.spmatrix
        abstract_property_specificity_limits = {"is_dense": False}

        @classmethod
        def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
            cls._validate_abstract_props(props)
            nrows, ncols = obj.shape
            is_square = nrows == ncols
            dtype = dtypes.dtypes_simplified[obj.dtype]

            ret = dict(dtype=dtype, is_square=is_square)

            # slow properties, only compute if asked
            for propname in prop:
                if propname == "is_symmetric":
                    ret["is_symmetric"] = is_square and (obj.T != obj).nnz == 0

            return ret

        @classmethod
        def compare_objects(
            cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True
        ):
            if not isinstance(obj1, cls.value_type) or not isinstance(
                obj2, cls.value_type
            ):
                raise TypeError("objects must be scipy.spmatrix")

            if check_values and obj1.dtype != obj2.dtype:
                return False
            if obj1.shape != obj2.shape:
                return False
            if issubclass(obj1.dtype.type, np.floating) or not check_values:
                d1 = obj1.tocsr()
                d2 = obj2.tocsr()
                if not (d1.indptr == d2.indptr).all():
                    return False
                if not (d1.indices == d2.indices).all():
                    return False
                if check_values:
                    return np.isclose(
                        d1.data, d2.data, rtol=rel_tol, atol=abs_tol
                    ).all()
                else:
                    return True
            else:
                return (obj1 != obj2).nnz == 0

    class ScipyAdjacencyMatrix(Wrapper, abstract=Graph):
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
        def num_nodes(self):
            return self.value.shape[0]

        @property
        def node_index(self):
            if self._node_index is None:
                self._node_index = SequentialNodes(self.value.shape[0])
            return self._node_index

        @property
        def format(self):
            return self.value.format

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
                # Yes this is weird, but it seems to be the only way to do it; one axis at a time
                data = data.tocsr()[index_converter][:, index_converter]
            return ScipyAdjacencyMatrix(
                data,
                weights=self._weights,
                is_directed=self._is_directed,
                node_index=node_index,
                transposed=self.transposed,
            )

        @classmethod
        def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
            cls._validate_abstract_props(props)
            return dict(
                dtype=obj._dtype, weights=obj._weights, is_directed=obj._is_directed
            )

        @classmethod
        def compare_objects(
            cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True
        ):
            if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
                raise TypeError("objects must be ScipyAdjacencyMatrix")

            if obj1.num_nodes != obj2.num_nodes:
                return False
            if obj1.value.nnz != obj2.value.nnz:
                return False
            if check_values and (
                obj1._dtype != obj2._dtype
                or obj1._weights != obj2._weights
                or obj1._is_directed != obj2._is_directed
            ):
                return False
            # Convert to a common node indexing scheme
            try:
                obj2 = obj2.rebuild_for_node_index(obj1.node_index)
            except ValueError:
                return False
            # Handle transposed states
            d1 = obj1.value.T if obj1.transposed else obj1.value
            d2 = obj2.value.T if obj2.transposed else obj2.value
            # Compare
            d1 = d1.tocsr()
            d2 = d2.tocsr()
            if not (d1.indptr == d2.indptr).all():
                return False
            # Ensure sorted indices for numpy matching to work
            d1.sort_indices()
            d2.sort_indices()
            if not (d1.indices == d2.indices).all():
                return False
            if check_values and obj1._weights != "unweighted":
                if issubclass(d1.dtype.type, np.floating):
                    return np.isclose(
                        d1.data, d2.data, rtol=rel_tol, atol=abs_tol
                    ).all()
                else:
                    return (d1.data == d2.data).all()
            return True
