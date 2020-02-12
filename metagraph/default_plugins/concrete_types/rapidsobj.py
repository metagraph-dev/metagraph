from ... import PluginRegistry, ConcreteType
from ..abstract_types import DataFrameType, GraphType, WeightedGraphType

reg = PluginRegistry("metagraph_core")

try:
    import cugraph
except ImportError:
    cugraph = None

try:
    import cudf
except ImportError:
    cudf = None


if cugraph is not None:

    # class CuGraph:
    #     def __init__(self, graph):
    #         self.obj = graph
    #         assert isinstance(graph, cugraph.DiGraph)

    class CuGraphWeighted:
        def __init__(self, graph, weight_label="weight"):
            self.obj = graph
            self.weight_label = weight_label
            assert isinstance(graph, cugraph.DiGraph)
            assert (
                weight_label in graph.nodes(data=True)[0]
            ), f"Graph is missing specified weight label: {weight_label}"

    @reg.register
    class CuGraphType(ConcreteType):
        name = "CuGraph"
        abstract = GraphType
        value_class = cugraph.DiGraph

    @reg.register
    class CuGraphWeighted(ConcreteType):
        name = "CuGraphWeighted"
        abstract = WeightedGraphType
        value_class = CuGraphWeighted


if cudf is not None:

    # class CuDataFrame:
    #     def __init__(self, df):
    #         self.obj = df
    #         assert isinstance(df, cudf.DataFrame)

    class CuDFEdgeList:
        def __init__(self, df, src_label="source", dest_label="destination"):
            self.obj = df
            self.src_label = src_label
            self.dest_label = dest_label
            assert isinstance(df, cudf.DataFrame)
            assert src_label in df, f"Indicated src_label not found: {src_label}"
            assert dest_label in df, f"Indicated dest_label not found: {dest_label}"

    class CuDFWeightedEdgeList(CuDFEdgeList):
        def __init__(
            self,
            df,
            src_label="source",
            dest_label="destination",
            weight_label="weight",
        ):
            super().__init__(df, src_label, dest_label)
            self.weight_label = weight_label
            assert isinstance(df, cudf.DataFrame)
            assert (
                weight_label in df
            ), f"Indicated weight_label not found: {weight_label}"

    @reg.register
    class CuDFType(ConcreteType):
        name = "CuDF"
        abstract = DataFrameType
        value_class = cudf.DataFrame

    @reg.register
    class CuDFEdgeListType(ConcreteType):
        name = "CuDFEdgeList"
        abstract = GraphType
        value_class = CuDFEdgeList

    @reg.register
    class CuDFWeightedEdgeList(ConcreteType):
        name = "CuDFWeightedEdgeList"
        abstract = WeightedGraphType
        value_class = CuDFWeightedEdgeList
