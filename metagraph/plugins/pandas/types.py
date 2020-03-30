from metagraph import ConcreteType, Wrapper, dtypes
from metagraph.types import DataFrame, Graph
from metagraph.plugins import has_pandas


if has_pandas:
    import pandas as pd

    class PandasDataFrameType(ConcreteType, abstract=DataFrame):
        value_type = pd.DataFrame

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
            is_directed=True,
        ):
            """
            Create a new Graph represented by a PandasEdgeList

            :param df:
            :param src_label:
            :param dst_label:
            :param weight_label:
            :param is_directed: If False, assumes edges are only represented once, not twice
            """
            self.value = df
            self.is_directed = is_directed
            self.src_label = src_label
            self.dst_label = dst_label
            self.weight_label = weight_label
            self._assert_instance(df, pd.DataFrame)
            self._assert(src_label in df, f"Indicated src_label not found: {src_label}")
            self._assert(dst_label in df, f"Indicated dst_label not found: {dst_label}")
            if weight_label is not None:
                self._assert(
                    weight_label in df,
                    f"Indicated weight_label not found: {weight_label}",
                )

        @classmethod
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                if obj.weight_label is None:
                    dtype = "bool"
                    weights = "unweighted"
                else:
                    values = obj.value[obj.weight_label]
                    dtype = dtypes.dtypes_simplified[values.dtype]
                    if dtype == "str":
                        weights = "any"
                    elif dtype == "bool":
                        if values.all():
                            weights = "unweighted"
                        else:
                            weights = "non-negative"
                    else:
                        min_val = values.min()
                        if min_val < 0:
                            weights = "any"
                        elif min_val == 0:
                            weights = "non-negative"
                        else:
                            if dtype == "int" and min_val == 1 and values.max() == 1:
                                weights = "unweighted"
                            else:
                                weights = "positive"
                ret_val.abstract_instance = Graph(
                    dtype=dtype, weights=weights, is_directed=obj.is_directed
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")
