from ... import ConcreteType, Wrapper
from ..abstract_types import DataFrame, Graph, WeightedGraph
from .. import registry, cugraph, cudf


if cugraph is not None:

    @registry.register
    class CuGraphType(ConcreteType, abstract=Graph):
        value_type = cugraph.DiGraph

    @registry.register
    class CuGraphWeighted(Wrapper, abstract=WeightedGraph):
        def __init__(self, graph, weight_label="weight"):
            self.value = graph
            self.weight_label = weight_label
            self._assert_instance(graph, cugraph.DiGraph)
            self._assert(
                weight_label in graph.nodes(data=True)[0],
                f"Graph is missing specified weight label: {weight_label}",
            )


if cudf is not None:

    @registry.register
    class CuDFType(ConcreteType, abstract=DataFrame):
        value_class = cudf.DataFrame

    @registry.register
    class CuDFEdgeList(Wrapper, abstract=Graph):
        def __init__(self, df, src_label="source", dest_label="destination"):
            self.value = df
            self.src_label = src_label
            self.dest_label = dest_label
            self._assert_instance(df, cudf.DataFrame)
            self._assert(src_label in df, f"Indicated src_label not found: {src_label}")
            self._assert(
                dest_label in df, f"Indicated dest_label not found: {dest_label}"
            )

    @registry.register
    class CuDFWeightedEdgeList(CuDFEdgeList, abstract=WeightedGraph):
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
