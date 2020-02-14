from ... import ConcreteType, Wrapper
from ..abstract_types import DataFrame, Graph, WeightedGraph
from .. import registry


try:
    import pandas as pd
except ImportError:
    pd = None


if pd is not None:

    @registry.register
    class PandasDataFrameType(ConcreteType, abstract=DataFrame):
        value_type = pd.DataFrame

    @registry.register
    class PandasEdgeList(Wrapper, abstract=Graph):
        def __init__(self, df, src_label="source", dest_label="destination"):
            self.value = df
            self.src_label = src_label
            self.dest_label = dest_label
            assert isinstance(df, pd.DataFrame)
            assert src_label in df, f"Indicated src_label not found: {src_label}"
            assert dest_label in df, f"Indicated dest_label not found: {dest_label}"

    @registry.register
    class PandasWeightedEdgeList(PandasEdgeList, abstract=WeightedGraph):
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
