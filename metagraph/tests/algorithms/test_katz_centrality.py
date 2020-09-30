from metagraph.tests.util import default_plugin_resolver
import networkx as nx
import numpy as np
from . import MultiVerify


def test_katz_centrality(default_plugin_resolver):
    r"""
              +-+
     ------>  |1| ----------------------------
     |        +-+                            |
     |                                       |
     |         |                             |
     |         v                             |
                                             V
    +-+  <--  +-+       +-+       +-+       +-+
    |0|       |2|  <--  |3|  -->  |4|  <--  |5|
    +-+  -->  +-+       +-+       +-+       +-+
    """
    dpr = default_plugin_resolver
    networkx_graph_data = [
        (0, 1),
        (0, 2),
        (2, 0),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 4),
        (5, 4),
    ]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_edges_from(networkx_graph_data)
    data = {
        0: 0.4069549895218489,
        1: 0.40687482321632046,
        2: 0.41497162410274485,
        3: 0.40280527348222406,
        4: 0.410902066312543,
        5: 0.4068740216338262,
    }
    expected_val = dpr.wrappers.NodeMap.PythonNodeMap(data)
    graph = dpr.wrappers.Graph.NetworkXGraph(networkx_graph)
    MultiVerify(dpr).compute("centrality.katz", graph, tolerance=1e-7).assert_equal(
        expected_val, rel_tol=1e-5
    )
