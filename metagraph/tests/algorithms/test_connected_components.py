import networkx as nx
from . import MultiVerify


def test_connected_components(default_plugin_resolver):
    """
0 ---2-- 1        5 --10-- 6
|      / |        |      / 
|     /  |        |     /   
1   7    3        5   11   
|  /     |        |  /    
| /      |        | /      
3 --8--- 4        2 --6--- 7
    """
    dpr = default_plugin_resolver
    ebunch = [
        (0, 3, 1),
        (1, 0, 2),
        (1, 4, 3),
        (2, 5, 5),
        (2, 7, 6),
        (3, 1, 7),
        (3, 4, 8),
        (5, 6, 10),
        (6, 2, 11),
    ]
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrapper.Graph.NetworkXGraph(nx_graph)
    expected_answer_unwrapped = {0: 0, 1: 0, 3: 0, 4: 0, 2: 1, 5: 1, 6: 1, 7: 1}
    expected_answer = dpr.wrapper.Nodes.PythonNodes(expected_answer_unwrapped)
    MultiVerify(dpr, "clustering.connected_components", graph).assert_equals(
        expected_answer
    )


def test_strongly_connected_components(default_plugin_resolver):
    """
          +-+
 ----9->  |1|
 |        +-+
 | 
 |         |
 |         6
 |         |
 |         v

+-+  <-7-  +-+        +-+
|0|        |2|  <-5-  |3|
+-+  -8->  +-+        +-+
"""
    dpr = default_plugin_resolver
    networkx_graph_data = [
        (0, 1, 9),
        (0, 2, 8),
        (2, 0, 7),
        (1, 2, 6),
        (3, 2, 5),
    ]
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(networkx_graph_data, weight="weight")
    graph = dpr.wrapper.Graph.NetworkXGraph(nx_graph)
    expected_answer_unwrapped = {0: 0, 1: 0, 2: 0, 3: 1}
    expected_answer = dpr.wrapper.Nodes.PythonNodes(expected_answer_unwrapped)
    MultiVerify(dpr, "clustering.strongly_connected_components", graph).assert_equals(
        expected_answer
    )
