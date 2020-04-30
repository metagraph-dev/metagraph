from . import MultiVerify
import networkx as nx
import numpy as np


def test_bfs(default_plugin_resolver):
    """
0 <--2-- 1        5 --10-> 6
|        |      ^ ^      / 
|        |     /  |     /   
1        3    9   5   11   
|        |  /     |   /    
v        v /        v      
3 --8--> 4 <--4-- 2 --6--> 7
    """
    dpr = default_plugin_resolver
    ebunch = [
        (0, 3, 1),
        (1, 0, 2),
        (1, 4, 3),
        (2, 4, 4),
        (2, 5, 5),
        (3, 4, 8),
        (4, 5, 9),
        (5, 6, 10),
        (6, 2, 11),
    ]
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrapper.Graph.NetworkXGraph(nx_graph)
    correct_answer = dpr.wrapper.Vector.NumpyVector(np.array([0, 3, 4, 5, 6, 2]))
    MultiVerify(dpr, "traversal.breadth_first_search", graph, 0).assert_equals(
        correct_answer
    )
