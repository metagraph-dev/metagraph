from metagraph.tests.util import default_plugin_resolver
import networkx as nx
from . import MultiVerify


def test_extract_edgemap(default_plugin_resolver):
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
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(
        [(1, 0, 2), (1, 4, 3), (2, 5, 5), (2, 7, 6), (3, 1, 7), (5, 6, 10), (6, 2, 11),]
    )
    desired_nodes = {2, 5, 6}
    nx_extracted_edgemap_graph = nx.Graph()
    nx_extracted_edgemap_graph.add_weighted_edges_from(
        [(2, 5, 5), (5, 6, 10), (6, 2, 11)]
    )
    graph = dpr.wrappers.EdgeMap.NetworkXEdgeMap(nx_graph)
    desired_nodes_wrapped = dpr.wrappers.NodeSet.PythonNodeSet(desired_nodes)
    extract_edgemap_graph = dpr.wrappers.EdgeMap.NetworkXEdgeMap(
        nx_extracted_edgemap_graph
    )
    MultiVerify(
        dpr, "subgraph.extract_edgemap", graph, desired_nodes_wrapped
    ).assert_equals(extract_edgemap_graph)


def test_extract_edgeset(default_plugin_resolver):
    """
0 ----- 1        5 ----- 6
      / |        |     / 
     /  |        |    /   
    /   |        |   /   
   /    |        |  /    
  /     |        | /      
3       4        2 ----- 7
    """
    dpr = default_plugin_resolver
    nx_graph = nx.Graph()
    nx_graph.add_edges_from([(1, 0), (1, 4), (2, 5), (2, 7), (3, 1), (5, 6), (6, 2)])
    desired_nodes = {2, 5, 6}
    nx_extracted_edgeset_graph = nx.Graph()
    nx_extracted_edgeset_graph.add_edges_from([(2, 5), (5, 6), (6, 2)])
    graph = dpr.wrappers.EdgeSet.NetworkXEdgeSet(nx_graph)
    desired_nodes_wrapped = dpr.wrappers.NodeSet.PythonNodeSet(desired_nodes)
    extract_edgeset_graph = dpr.wrappers.EdgeSet.NetworkXEdgeSet(
        nx_extracted_edgeset_graph
    )
    MultiVerify(
        dpr, "subgraph.extract_edgeset", graph, desired_nodes_wrapped
    ).assert_equals(extract_edgeset_graph)


def test_k_core(default_plugin_resolver):
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
    graph = dpr.wrappers.EdgeMap.NetworkXEdgeMap(nx_graph)
    k_core_graph = dpr.wrappers.EdgeMap.NetworkXEdgeMap(nx_k_core_graph)
    MultiVerify(dpr, "subgraph.k_core", graph, k).assert_equals(k_core_graph)


def test_k_core_unweighted(default_plugin_resolver):
    """
0 ----- 1        5 ----- 6
      / |        |     / 
     /  |        |    /   
    /   |        |   /   
   /    |        |  /    
  /     |        | /      
3       4        2 ----- 7
    """
    dpr = default_plugin_resolver
    k = 2
    nx_graph = nx.Graph()
    nx_graph.add_edges_from([(1, 0), (1, 4), (2, 5), (2, 7), (3, 1), (5, 6), (6, 2)])
    nx_k_core_graph = nx.Graph()
    nx_k_core_graph.add_edges_from([(2, 5), (5, 6), (6, 2)])
    graph = dpr.wrappers.EdgeSet.NetworkXEdgeSet(nx_graph)
    k_core_graph = dpr.wrappers.EdgeSet.NetworkXEdgeSet(nx_k_core_graph)
    MultiVerify(dpr, "subgraph.k_core_unweighted", graph, k).assert_equals(k_core_graph)
