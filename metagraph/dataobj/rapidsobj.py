from .base import Graph, WeightedGraph, DataFrame, EdgeListDF, WeightedEdgeListDF

try:
    import cugraph
except ImportError:
    cugraph = None

try:
    import cudf
except ImportError:
    cudf = None


if cugraph is not None:

    class CuGraph(Graph):
        def __init__(self, graph):
            self.obj = graph
            assert isinstance(graph, cugraph.DiGraph)

    class CuGraphWeighted(WeightedGraph):
        def __init__(self, graph, weight_label="weight"):
            self.obj = graph
            self.weight_label = weight_label
            assert isinstance(graph, cugraph.DiGraph)
            assert (
                weight_label in graph.nodes(data=True)[0]
            ), f"Graph is missing specified weight label: {weight_label}"


if cudf is not None:

    class CuDataFrame(DataFrame):
        def __init__(self, df):
            self.obj = df
            assert isinstance(df, cudf.DataFrame)

    class PandasEdgeListDF(EdgeListDF):
        def __init__(self, df, source_label="sources", dest_label="destination"):
            super().__init__(source_label, dest_label)
            self.obj = df
            assert isinstance(df, cudf.DataFrame)
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
            assert isinstance(df, cudf.DataFrame)
            assert (
                source_label in df
            ), f"Indicated source_label not found: {source_label}"
            assert dest_label in df, f"Indicated dest_label not found: {dest_label}"
            assert (
                weight_label in df
            ), f"Indicated weight_label not found: {weight_label}"
