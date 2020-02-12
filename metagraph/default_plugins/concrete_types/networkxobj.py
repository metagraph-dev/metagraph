from ... import PluginRegistry, ConcreteType
from ..abstract_types import GraphType, WeightedGraphType

reg = PluginRegistry("metagraph_core")

try:
    import networkx as nx
except ImportError:
    nx = None


if nx is not None:

    # class NetworkXGraph:
    #     def __init__(self, graph):
    #         self.obj = graph
    #         assert isinstance(graph, nx.DiGraph)

    class NetworkXWeightedGraph:
        def __init__(self, graph, weight_label="weight"):
            self.obj = graph
            self.weight_label = weight_label
            assert isinstance(graph, nx.DiGraph)
            assert (
                weight_label in graph.nodes(data=True)[0]
            ), f"Graph is missing specified weight label: {weight_label}"

    @reg.register
    class NetworkXGraphType(ConcreteType):
        name = "NetworkXGraph"
        abstract = GraphType
        value_class = nx.DiGraph

    @reg.register
    class NetworkXWeightedGraph(ConcreteType):
        name = "NetowrkXWeightedGraph"
        abstract = WeightedGraphType
        value_class = NetworkXWeightedGraph
