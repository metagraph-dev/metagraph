from metagraph.tests.util import default_plugin_resolver
import networkx as nx
from . import MultiVerify


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
