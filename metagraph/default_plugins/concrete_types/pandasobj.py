from ... import PluginRegistry, ConcreteType
from ..abstract_types import DataFrameType, GraphType, WeightedGraphType

reg = PluginRegistry("metagraph_core")

try:
    import pandas as pd
except ImportError:
    pd = None


if pd is not None:

    # class PandasDataFrame:
    #     def __init__(self, df):
    #         self.obj = df
    #         assert isinstance(df, pd.DataFrame)

    class PandasEdgeList:
        def __init__(self, df, src_label="source", dest_label="destination"):
            self.obj = df
            self.src_label = src_label
            self.dest_label = dest_label
            assert isinstance(df, pd.DataFrame)
            assert src_label in df, f"Indicated src_label not found: {src_label}"
            assert dest_label in df, f"Indicated dest_label not found: {dest_label}"

    class PandasWeightedEdgeList(PandasEdgeList):
        def __init__(
            self,
            df,
            src_label="source",
            dest_label="destination",
            weight_label="weight",
        ):
            super().__init__(df, src_label, dest_label)
            self.weight_label = weight_label
            assert (
                weight_label in df
            ), f"Indicated weight_label not found: {weight_label}"

    @reg.register
    class PandasDataFrameType(ConcreteType):
        name = "PandasDataFrame"
        abstract = DataFrameType
        value_class = pd.DataFrame

    @reg.register
    class PandasEdgeListType(ConcreteType):
        name = "PandasEdgeList"
        abstract = GraphType
        value_class = PandasEdgeList

    @reg.register
    class PandasWeightedEdgeList(ConcreteType):
        name = "PandasWeightedEdgeList"
        abstract = WeightedGraphType
        value_class = PandasWeightedEdgeList
