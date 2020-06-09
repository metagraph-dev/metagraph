import numpy as np
from typing import List, Dict, Any
from metagraph import ConcreteType, dtypes
from metagraph.types import DataFrame, EdgeSet, EdgeMap
from metagraph.wrappers import EdgeSetWrapper, EdgeMapWrapper
from metagraph.plugins import has_pandas
import math


if has_pandas:
    import pandas as pd

    class PandasDataFrameType(ConcreteType, abstract=DataFrame):
        value_type = pd.DataFrame

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            digits_precision = round(-math.log(rel_tol, 10))
            pd.testing.assert_frame_equal(
                obj1, obj2, check_like=True, check_less_precise=digits_precision
            )

    class PandasEdgeSet(EdgeSetWrapper, abstract=EdgeSet):
        def __init__(
            self, df, src_label="source", dst_label="target", *, is_directed=True
        ):
            self._assert_instance(df, pd.DataFrame)
            self.value = df
            self.is_directed = is_directed
            self.src_label = src_label
            self.dst_label = dst_label
            self._assert(src_label in df, f"Indicated src_label not found: {src_label}")
            self._assert(dst_label in df, f"Indicated dst_label not found: {dst_label}")
            # Build the MultiIndex representing the edges
            self.index = df.set_index([src_label, dst_label]).index

        @property
        def num_nodes(self):
            src_nodes, dst_nodes = self.index.levels
            return len(src_nodes | dst_nodes)

        @classmethod
        def assert_equal(
            cls, obj1, obj2, props1, props2, *, rel_tol=None, abs_tol=None
        ):
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            g1 = obj1.value
            g2 = obj2.value
            assert len(g1) == len(g2), f"{len(g1)} != {len(g2)}"
            assert len(obj1.index & obj2.index) == len(
                obj1.index
            ), f"{len(obj1.index & obj2.index)} != {len(obj1.index)}"

    class PandasEdgeMap(EdgeMapWrapper, abstract=EdgeMap):
        """
        Graph represented as a pandas DataFrame with edges indicated by source and destination columns
        """

        def __init__(
            self,
            df,
            src_label="source",
            dst_label="target",
            weight_label="weight",
            *,
            is_directed=True,
        ):
            """
            Create a new EdgeMap represented by a weighted edge list

            :param df:
            :param src_label:
            :param dst_label:
            :param weight_label:
            :param is_directed: If False, assumes edges are bidirectional; duplicate edges with different weights
                                          will raise an error
            :param node_label:
            """
            self._assert_instance(df, pd.DataFrame)
            self.value = df
            self.is_directed = is_directed
            self.src_label = src_label
            self.dst_label = dst_label
            self.weight_label = weight_label
            self._assert(src_label in df, f"Indicated src_label not found: {src_label}")
            self._assert(dst_label in df, f"Indicated dst_label not found: {dst_label}")
            self._assert(
                weight_label in df, f"Indicated weight_label not found: {weight_label}"
            )
            # Build the MultiIndex representing the edges
            self.index = df.set_index([src_label, dst_label]).index

        @property
        def num_nodes(self):
            src_nodes, dst_nodes = self.index.levels
            return len(src_nodes | dst_nodes)

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"is_directed", "dtype"} - ret.keys():
                if prop == "is_directed":
                    ret[prop] = obj.is_directed
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[
                        obj.value[obj.weight_label].dtype
                    ]

            # slow properties, only compute if asked
            for prop in props - ret.keys():
                if prop == "weights":
                    if ret["dtype"] == "str":
                        weights = "any"
                    elif ret["dtype"] == "bool":
                        weights = "non-negative"
                    else:
                        min_val = obj.value[obj.weight_label].min()
                        if min_val < 0:
                            weights = "any"
                        elif min_val == 0:
                            weights = "non-negative"
                        else:
                            weights = "positive"
                    ret[prop] = weights

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            g1 = obj1.value
            g2 = obj2.value
            assert len(g1) == len(g2), f"{len(g1)} != {len(g2)}"
            assert len(obj1.index & obj2.index) == len(
                obj1.index
            ), f"{len(obj1.index & obj2.index)} != {len(obj1.index)}"
            # Ensure dataframes are indexed the same
            if not (obj1.index == obj2.index).all():
                g2 = g2.set_index(obj2.index).reindex(obj1.index).reset_index(drop=True)
            # Compare
            v1 = g1[obj1.weight_label]
            v2 = g2[obj2.weight_label]
            if issubclass(v1.dtype.type, np.floating):
                assert np.isclose(v1, v2, rtol=rel_tol, atol=abs_tol).all()
            else:
                assert (v1 == v2).all()
