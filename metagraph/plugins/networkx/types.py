from metagraph import ConcreteType, Wrapper
from metagraph.types import Graph, WeightedGraph
from metagraph.plugins import has_networkx


if has_networkx:
    import networkx as nx

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
