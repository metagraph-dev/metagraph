import networkx as nx
from . import MultiVerify


def test_extract_subgraph(default_plugin_resolver):
    """
          +-+
 ------>  |1|
 |        +-+
 | 
 |         |
 |         v

+-+  <--  +-+       +-+
|0|       |2|  <--  |3|
+-+  -->  +-+       +-+
"""
    dpr = default_plugin_resolver
    networkx_graph_data = [
        (0, 1),
        (0, 2),
        (2, 0),
        (1, 2),
        (3, 2),
    ]
    subgraph_nodes = [0, 2, 3]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_edges_from(networkx_graph_data)
    networkx_subgraph = nx.subgraph(networkx_graph, subgraph_nodes)
    networkx_subgraph_wrapped = dpr.wrapper.Graph.NetworkXGraph(networkx_subgraph)
    graph = dpr.wrapper.Graph.NetworkXGraph(networkx_graph, dtype="int")
    MultiVerify(dpr, "subgraph.extract_subgraph", graph, subgraph_nodes).assert_equals(
        networkx_subgraph_wrapped
    )
