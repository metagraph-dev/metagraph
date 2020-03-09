from metagraph import ConcreteType, Wrapper
from metagraph.types import DataFrame, Graph, WeightedGraph
from metagraph.plugins import has_pandas


if has_pandas:
    import pandas as pd

    class PandasDataFrameType(ConcreteType, abstract=DataFrame):
        value_type = pd.DataFrame

    class PandasEdgeList(Wrapper, abstract=Graph):
        """
        Graph represented as a pandas DataFrame with edges indicated by source and destination columns
        """

        def __init__(self, df, src_label="source", dest_label="destination"):
            self.value = df
            self.src_label = src_label
            self.dest_label = dest_label
            self._assert_instance(df, pd.DataFrame)
            self._assert(src_label in df, f"Indicated src_label not found: {src_label}")
            self._assert(
                dest_label in df, f"Indicated dest_label not found: {dest_label}"
            )

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
            self._assert(
                weight_label in df, f"Indicated weight_label not found: {weight_label}"
            )
