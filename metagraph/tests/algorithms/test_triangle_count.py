from metagraph.tests.util import default_plugin_resolver
import networkx as nx
from . import MultiVerify

# Simple graph with 5 triangles
# 0 - 1    5 - 6
# | X |    | /
# 3 - 4 -- 2 - 7
simple_graph_data = [
    [0, 1, 100],
    [0, 3, 200],
    [0, 4, 300],
    [1, 3, 50],
    [1, 4, 55],
    [2, 4, 60],
    [2, 5, 65],
    [2, 6, 70],
    [3, 4, 75],
    [5, 6, 20],
    [6, 7, 10],
]


def test_triangle_count(default_plugin_resolver):
    dpr = default_plugin_resolver
    # Build simple graph with 5 triangles
    simple_graph = nx.Graph()
    simple_graph.add_weighted_edges_from(simple_graph_data)
    # Convert to wrapper
    graph = dpr.wrappers.Graph.NetworkXGraph(simple_graph)

    MultiVerify(dpr).compute("cluster.triangle_count", graph).assert_equal(5)
