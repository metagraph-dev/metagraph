from ... import ConcreteType, Wrapper
from ..abstract_types import Graph, WeightedGraph
from .. import networkx


if networkx is not None:
    nx = networkx

    class NetworkXGraphType(ConcreteType, abstract=Graph):
        value_type = nx.DiGraph

    class NetworkXWeightedGraph(Wrapper, abstract=WeightedGraph):
        def __init__(self, graph, weight_label="weight"):
            self.value = graph
            self.weight_label = weight_label
            self._assert_instance(graph, nx.DiGraph)
            if weight_label not in graph.nodes(data=True)[0]:
                raise TypeError(
                    f"Graph is missing specified weight label: {weight_label}"
                )
