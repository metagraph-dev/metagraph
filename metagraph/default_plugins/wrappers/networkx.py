from ... import ConcreteType, Wrapper
from ..abstract_types import Graph, WeightedGraph
from .. import registry, networkx


if networkx is not None:
    nx = networkx

    @registry.register
    class NetworkXGraphType(ConcreteType, abstract=Graph):
        value_type = nx.DiGraph

    @registry.register
    class NetworkXWeightedGraph(Wrapper, abstract=WeightedGraph):
        def __init__(self, graph, weight_label="weight"):
            self.value = graph
            self.weight_label = weight_label
            assert isinstance(graph, nx.DiGraph)
            assert (
                weight_label in graph.nodes(data=True)[0]
            ), f"Graph is missing specified weight label: {weight_label}"
