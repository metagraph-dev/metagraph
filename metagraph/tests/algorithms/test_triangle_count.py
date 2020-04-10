import networkx as nx
from . import apply_all, verify_all

# Simple graph with 5 triangles
# 0 - 1    5 - 6
# | X |    | /
# 3 - 4 -- 2 - 7
simple_graph_data = [
    [0, 1],
    [0, 3],
    [0, 4],
    [1, 3],
    [1, 4],
    [2, 4],
    [2, 5],
    [2, 6],
    [3, 4],
    [5, 6],
    [6, 7],
]


def test_triangle_count(default_plugin_resolver):
    dpr = default_plugin_resolver
    # Build simple graph with 5 triangles
    simple_graph = nx.Graph()
    simple_graph.add_edges_from(simple_graph_data)
    # Convert to wrapper
    graph = dpr.wrapper.Graph.NetworkXGraph(simple_graph)

    algo_results = apply_all(dpr, "cluster.triangle_count", graph)
    verify_all(dpr, 5, algo_results)
