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
    networkx_graph_data = [
        (0, 1),
        (0, 2),
        (2, 0),
        (1, 2),
        (3, 2),
    ]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_edges_from(networkx_graph_data)
    networkx_pagerank = nx.pagerank(networkx_graph, alpha=0.85, max_iter=50, tol=1e-05)
    networkx_pagerank_wrapped = dpr.wrapper.Nodes.PythonNodes(networkx_pagerank)
    graph = dpr.wrapper.Graph.NetworkXGraph(networkx_graph, dtype="int")
    MultiVerify(dpr, "link_analysis.pagerank", graph).assert_equals(
        networkx_pagerank_wrapped
    )
