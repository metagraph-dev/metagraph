import networkx as nx
from . import MultiVerify


def test_betweenness_centrality(default_plugin_resolver):
    """
0 <--2-- 1        5 --10-> 6
|      ^ |      ^ ^      / 
|     /  |     /  |     /   
1    7   3    9   5   11   
|   /    |  /     |   /    
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
        (2, 7, 6),
        (3, 1, 7),
        (3, 4, 8),
        (4, 5, 9),
        (5, 6, 10),
        (6, 2, 11),
    ]
    nx_graph = nx.DiGraph()
    nx_graph.add_weighted_edges_from(ebunch)
    graph = dpr.wrapper.Graph.NetworkXGraph(nx_graph)
    k = 8
    enable_normalization = False
    include_endpoints = False
    expected_answer_unwrapped = {
        0: 1.0,
        1: 1.0,
        2: 9.0,
        3: 6.0,
        4: 12.0,
        5: 13.0,
        6: 11.0,
        7: 0.0,
    }
    expected_answer = dpr.wrapper.Nodes.PythonNodes(
        expected_answer_unwrapped, dtype="float"
    )
    MultiVerify(
        dpr,
        "vertex_ranking.betweenness_centrality",
        graph,
        k,
        enable_normalization,
        include_endpoints,
    ).assert_equals(expected_answer)
