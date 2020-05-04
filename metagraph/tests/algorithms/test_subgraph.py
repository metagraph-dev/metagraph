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
    subgraph_nodes = [0, 2, 3]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_edges_from(
        [(0, 1), (0, 2), (2, 0), (1, 2), (3, 2),]
    )
    networkx_subgraph = nx.DiGraph()
    networkx_subgraph.add_edges_from(
        [(0, 2), (2, 0), (3, 2),]
    )
    networkx_subgraph_wrapped = dpr.wrapper.Graph.NetworkXGraph(networkx_subgraph)
    graph = dpr.wrapper.Graph.NetworkXGraph(
        networkx_graph, dtype="int", weight_label="weight"
    )
    MultiVerify(dpr, "subgraph.extract_subgraph", graph, subgraph_nodes).assert_equals(
        networkx_subgraph_wrapped
    )


def test_extract_subgraph(default_plugin_resolver):
    """
0 ---2-- 1        5 --10-- 6
       / |        |      / 
      /  |        |     /   
    7    3        5   11   
   /     |        |  /    
  /      |        | /      
3        4        2 --6--- 7
    """
    dpr = default_plugin_resolver
    k = 2
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(
        [(1, 0, 2), (1, 4, 3), (2, 5, 5), (2, 7, 6), (3, 1, 7), (5, 6, 10), (6, 2, 11),]
    )
    nx_k_core_graph = nx.Graph()
    nx_k_core_graph.add_weighted_edges_from(
        [(2, 5, 5), (5, 6, 10), (6, 2, 11),]
    )
    graph = dpr.wrapper.Graph.NetworkXGraph(nx_graph, weight_label="weight")
    k_core_graph = dpr.wrapper.Graph.NetworkXGraph(
        nx_k_core_graph, weight_label="weight"
    )
    MultiVerify(dpr, "subgraph.k_core", graph, k).assert_equals(k_core_graph)
