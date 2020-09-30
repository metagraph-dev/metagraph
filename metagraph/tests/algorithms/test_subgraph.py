from metagraph.tests.util import default_plugin_resolver
import networkx as nx
from . import MultiVerify


def test_extract_graph(default_plugin_resolver):
    r"""
    0 ---2-- 1        5 --10-- 6
           / |        |      /
          /  |        |     /
        7    3        5   11
       /     |        |  /
      /      |        | /
    3        4        2 --6--- 7
    """
    dpr = default_plugin_resolver
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(
        [(1, 0, 2), (1, 4, 3), (2, 5, 5), (2, 7, 6), (3, 1, 7), (5, 6, 10), (6, 2, 11),]
    )
    desired_nodes = {2, 5, 6}
    nx_extracted_graph = nx.Graph()
    nx_extracted_graph.add_weighted_edges_from([(2, 5, 5), (5, 6, 10), (6, 2, 11)])
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
    desired_nodes_wrapped = dpr.wrappers.NodeSet.PythonNodeSet(desired_nodes)
    extracted_graph = dpr.wrappers.Graph.NetworkXGraph(nx_extracted_graph)
    MultiVerify(dpr).compute(
        "subgraph.extract_subgraph", graph, desired_nodes_wrapped
    ).assert_equal(extracted_graph)


def test_k_core(default_plugin_resolver):
    r"""
    0 ---2-- 1        5 --10-- 6
           / |        |      /
          /  |        |     /
        7    3        5   11
       /     |        |  /
      /      |        | /
    3        4        2 --6--- 7
    """
    dpr = default_plugin_resolver
    k = 2
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(
        [(1, 0, 2), (1, 4, 3), (2, 5, 5), (2, 7, 6), (3, 1, 7), (5, 6, 10), (6, 2, 11),]
    )
    nx_k_core_graph = nx.Graph()
    nx_k_core_graph.add_weighted_edges_from(
        [(2, 5, 5), (5, 6, 10), (6, 2, 11),]
    )
    graph = dpr.wrappers.Graph.NetworkXGraph(nx_graph)
    k_core_graph = dpr.wrappers.Graph.NetworkXGraph(nx_k_core_graph)
    MultiVerify(dpr).compute("subgraph.k_core", graph, k).assert_equal(k_core_graph)
