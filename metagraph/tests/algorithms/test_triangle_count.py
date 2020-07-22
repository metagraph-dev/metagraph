from metagraph.tests.util import default_plugin_resolver
import networkx as nx
from . import MultiVerify

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
    graph = dpr.wrappers.Graph.NetworkXGraph(simple_graph)

    MultiVerify(dpr, "cluster.triangle_count", graph).assert_equals(5)
