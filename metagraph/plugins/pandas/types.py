import numpy as np
from metagraph import ConcreteType, Wrapper, dtypes, IndexedNodes
from metagraph.types import DataFrame, Graph, WEIGHT_CHOICES
from metagraph.plugins import has_pandas
import operator
import math


if has_pandas:
    import pandas as pd

    class PandasDataFrameType(ConcreteType, abstract=DataFrame):
        value_type = pd.DataFrame

        @classmethod
        def compare_objects(
            cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True
        ):
            if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
                raise TypeError("objects must be pandas DataFrames")

            if check_values:
                try:
                    digits_precision = round(-math.log(rel_tol, 10))
                    pd.testing.assert_frame_equal(
                        obj1, obj2, check_like=True, check_less_precise=digits_precision
                    )
                    return True
                except AssertionError:
                    return False
            else:
                return obj1.shape == obj2.shape

    class PandasEdgeList(Wrapper, abstract=Graph):
        """
        Graph represented as a pandas DataFrame with edges indicated by source and destination columns
        """

        def __init__(
            self,
            df,
            src_label="source",
            dst_label="target",
            weight_label=None,
            *,
            is_directed=True,
            weights=None,
            node_index=None,
        ):
            """
            Create a new Graph represented by a PandasEdgeList

            :param df:
            :param src_label:
            :param dst_label:
            :param weight_label:
            :param is_directed: If False, assumes edges are bidirectional; duplicate edges with different weights
                                          will raise an error
            :param node_index:
            """
            self._assert_instance(df, pd.DataFrame)
            self.value = df
            self.is_directed = is_directed
            self._node_index = node_index
            self.src_label = src_label
            self.dst_label = dst_label
            self.weight_label = weight_label
            self._assert(src_label in df, f"Indicated src_label not found: {src_label}")
            self._assert(dst_label in df, f"Indicated dst_label not found: {dst_label}")
            if weight_label is not None:
                self._assert(
                    weight_label in df,
                    f"Indicated weight_label not found: {weight_label}",
                )
            self._dtype = self._determine_dtype()
            self._weights = self._determine_weights(weights)
            # Build the MultiIndex representing the edges
            self.index = df.set_index([src_label, dst_label]).index

        def _determine_dtype(self):
            if self.weight_label is None:
                return "bool"

            values = self.value[self.weight_label]
            return dtypes.dtypes_simplified[values.dtype]

        def _determine_weights(self, weights):
            if weights is not None:
                if weights not in WEIGHT_CHOICES:
                    raise ValueError(f"Illegal weights: {weights}")
                return weights

            if self.weight_label is None:
                return "unweighted"
            if self._dtype == "str":
                return "any"
            values = self.value[self.weight_label]
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

        @property
        def num_nodes(self):
            src_nodes, dst_nodes = self.index.levels
            return len(src_nodes | dst_nodes)

        @property
        def node_index(self):
            if self._node_index is None:
                src_col = self.value[self.src_label]
                dst_col = self.value[self.dst_label]
                all_nodes = set(src_col.unique()) | set(dst_col.unique())
                if src_col.dtype == dtypes.int64 and dst_col.dtype == dtypes.int64:
                    all_nodes = sorted(all_nodes)
                self._node_index = IndexedNodes(all_nodes)
            return self._node_index

        @classmethod
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                ret_val.abstract_instance = Graph(
                    dtype=obj._dtype, weights=obj._weights, is_directed=obj.is_directed
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

        @classmethod
        def compare_objects(
            cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True
        ):
            if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
                raise TypeError("objects must be PandasEdgeList")

            if check_values and (
                obj1._dtype != obj2._dtype or obj1._weights != obj2._weights
            ):
                return False
            if obj1.is_directed != obj2.is_directed:
                return False
            g1 = obj1.value
            g2 = obj2.value
            if len(g1) != len(g2):
                return False
            if len(obj1.index & obj2.index) < len(obj1.index):
                return False
            # Ensure dataframes are indexed the same
            if not (obj1.index == obj2.index).all():
                g2 = g2.set_index(obj2.index).reindex(obj1.index).reset_index(drop=True)
            # Compare
            if check_values and obj1._weights != "unweighted":
                v1 = g1[obj1.weight_label]
                v2 = g2[obj2.weight_label]
                if issubclass(v1.dtype.type, np.floating):
                    return np.isclose(v1, v2, rtol=rel_tol, atol=abs_tol).all()
                else:
                    return (v1 == v2).all()
            return True
