import networkx as nx

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
    simple_graph = simple_graph.to_directed()

    # Convert to every concrete type to test all concrete algorithms
    for ct_name in dir(dpr.types.Graph):
        ct = getattr(dpr.types.Graph, ct_name)
        if ct == dpr.types.Graph.NetworkXGraphType:
            test_graph = simple_graph
        else:
            try:
                test_graph = dpr.translate(simple_graph, ct)
            except TypeError:
                continue

        test_algo = dpr.find_algorithm_exact("cluster.triangle_count", test_graph)
        if test_algo:
            num_triangles = test_algo(test_graph)
            assert isinstance(num_triangles, int), type(num_triangles)
            assert num_triangles == 5
