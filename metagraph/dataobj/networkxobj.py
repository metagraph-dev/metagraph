from .base import Graph, WeightedGraph

try:
    import networkx as nx
except ImportError:
    nx = None


if nx is not None:

    class NetworkXGraph(Graph):
        def __init__(self, graph):
            self.obj = graph
            assert isinstance(graph, nx.DiGraph)

    class NetworkXWeightedGraph(WeightedGraph):
        def __init__(self, graph, weight_label="weight"):
            self.obj = graph
            self.weight_label = weight_label
            assert isinstance(graph, nx.DiGraph)
            assert (
                weight_label in graph.nodes(data=True)[0]
            ), f"Graph is missing specified weight label: {weight_label}"
