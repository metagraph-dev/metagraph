from .base import DataFrame, EdgeListDF, WeightedEdgeListDF

try:
    import pandas as pd
except ImportError:
    pd = None


if pd is not None:

    class PandasDataFrame(DataFrame):
        def __init__(self, df):
            self.obj = df
            assert isinstance(df, pd.DataFrame)

    class PandasEdgeListDF(EdgeListDF):
        def __init__(self, df, source_label="sources", dest_label="destination"):
            super().__init__(source_label, dest_label)
            self.obj = df
            assert isinstance(df, pd.DataFrame)
            assert (
                source_label in df
            ), f"Indicated source_label not found: {source_label}"
            assert dest_label in df, f"Indicated dest_label not found: {dest_label}"

    class PandasWeightedEdgeListDF(WeightedEdgeListDF):
        def __init__(
            self,
            df,
            source_label="sources",
            dest_label="destination",
            weight_label="weights",
        ):
            super().__init__(source_label, dest_label, weight_label)
            self.obj = df
            assert isinstance(df, pd.DataFrame)
            assert (
                source_label in df
            ), f"Indicated source_label not found: {source_label}"
            assert dest_label in df, f"Indicated dest_label not found: {dest_label}"
            assert (
                weight_label in df
            ), f"Indicated weight_label not found: {weight_label}"
