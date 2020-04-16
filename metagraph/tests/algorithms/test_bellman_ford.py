import networkx as nx
from . import MultiVerify


# def test_bellman_ford(default_plugin_resolver):
#     """
# 0 <--2-- 1        5 --10-> 6
# |      ^ |      ^ ^      /
# |     /  |     /  |     /
# 1    7   3    9   5   11
# |   /    |  /     |   /
# v        v /        v
# 3 --8--> 4 <--4-- 2 --6--> 7
#     """
#     r = mg.resolver
#     ebunch = [
#         (0, 3, 1),
#         (1, 0, 2),
#         (1, 4, 3),
#         (2, 4, 4),
#         (2, 5, 5),
#         (2, 7, 6),
#         (3, 1, 7),
#         (3, 4, 8),
#         (4, 5, 9),
#         (5, 6, 10),
#         (6, 2, 11),
#     ]
#     nx_graph = nx.DiGraph()
#     nx_graph.add_weighted_edges_from(ebunch)
#     graph = dpr.wrapper.Graph.NetworkXGraph(nx_graph)
#     node_to_parent_mapping = {0: 0, 3: 0, 1: 3, 4: 3, 5: 4, 6: 5, 2: 6, 7: 2}
#     node_to_length_mapping = {0: 0, 3: 1, 1: 8, 4: 9, 5: 18, 6: 28, 2: 39, 7: 45}
#     expected_answer = (node_to_parent_mapping, node_to_length_mapping)
#     MultiVerify(dpr, "traversal.bellman_ford", graph, 0).assert_equals(expected_answer)
