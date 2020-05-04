import networkx as nx
import numpy as np
from . import MultiVerify


def test_pagerank(default_plugin_resolver):
    """
          +-+
 ------>  |1|
 |        +-+
 | 
 |         |
 |         v

+-+  <--  +-+       +-+
|0|       |2|  <--  |3|
+-+  -->  +-+       +-+
"""
    dpr = default_plugin_resolver
    networkx_graph_data = [(0, 1), (0, 2), (2, 0), (1, 2), (3, 2)]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_edges_from(networkx_graph_data)
    data = {
        0: 0.37252685132844066,
        1: 0.19582391181458728,
        2: 0.3941492368569718,
        3: 0.037500000000000006,
    }
    expected_val = dpr.wrapper.Nodes.PythonNodes(data)
    graph = dpr.wrapper.Graph.NetworkXGraph(networkx_graph, dtype="int")
    MultiVerify(dpr, "link_analysis.pagerank", graph, tolerance=1e-7).assert_equals(
        expected_val, rel_tol=1e-5
    )
