import pytest

grblas = pytest.importorskip("grblas")

from metagraph.tests.util import default_plugin_resolver
from metagraph.plugins.numpy.types import NumpyNodeSet
from metagraph.plugins.scipy.types import ScipyEdgeMap, ScipyEdgeSet, ScipyGraph
from metagraph.plugins.networkx.types import NetworkXGraph
from metagraph.plugins.graphblas.types import GrblasEdgeMap
from metagraph.plugins.pandas.types import PandasEdgeMap
from metagraph import NodeLabels
import networkx as nx
import scipy.sparse as ss
import pandas as pd
import numpy as np


def test_networkx_scipy_graph_from_edgemap(default_plugin_resolver):
    dpr = default_plugin_resolver
    g = nx.DiGraph()
    g.add_weighted_edges_from([(2, 2, 1), (2, 7, 2), (7, 7, 0), (7, 0, 3), (0, 7, 3)])
    x = NetworkXGraph(g)
    # Convert networkx -> scipy adjacency
    #    0 2 7
    # 0 [    3]
    # 2 [  1 2]
    # 7 [3   0]
    m = ss.coo_matrix(
        ([3, 1, 2, 3, 0], ([0, 1, 1, 2, 2], [2, 1, 2, 0, 2])), dtype=np.int64
    )
    intermediate = ScipyGraph(ScipyEdgeMap(m, [0, 2, 7]))
    y = dpr.translate(x, ScipyGraph)
    dpr.assert_equal(y, intermediate)


def test_networkx_scipy_graph_from_edgeset(default_plugin_resolver):
    dpr = default_plugin_resolver
    g = nx.DiGraph()
    g.add_edges_from([(2, 2), (2, 7), (7, 7), (7, 0), (0, 7)])
    x = NetworkXGraph(g)
    # Convert networkx -> scipy adjacency
    #    0 2 7
    # 0 [    1]
    # 2 [  1 1]
    # 7 [1   1]
    m = ss.coo_matrix(
        ([1, 1, 1, 1, 1], ([0, 1, 1, 2, 2], [2, 1, 2, 0, 2])), dtype=np.int64
    )
    intermediate = ScipyGraph(ScipyEdgeSet(m, [0, 2, 7]))
    y = dpr.translate(x, ScipyGraph)
    dpr.assert_equal(y, intermediate)


def test_scipy_graphblas_edgemap(default_plugin_resolver):
    dpr = default_plugin_resolver
    #    0 2 7
    # 0 [1 2  ]
    # 2 [  0 3]
    # 7 [  3  ]
    g = ss.coo_matrix(
        ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1])), dtype=np.int64
    )
    x = ScipyEdgeMap(g, [0, 2, 7])
    # Convert scipy adjacency to graphblas
    m = grblas.Matrix.from_values(
        [0, 0, 2, 2, 7], [0, 2, 2, 7, 2], [1, 2, 0, 3, 3], dtype=grblas.dtypes.INT64
    )
    intermediate = GrblasEdgeMap(m)
    y = dpr.translate(x, GrblasEdgeMap)
    dpr.assert_equal(y, intermediate)


# def test_networkx_2_pandas(default_plugin_resolver):
#     dpr = default_plugin_resolver
#     g = nx.DiGraph()
#     g.add_weighted_edges_from([(2, 2, 1), (2, 7, 2), (7, 7, 0), (7, 0, 3), (0, 7, 3)])
#     x = NetworkXGraph(g)
#     # Convert networkx -> pandas edge list
#     df = pd.DataFrame(
#         {
#             "source": [2, 2, 7, 0, 7],
#             "target": [2, 7, 0, 7, 7],
#             "weight": [1, 2, 3, 3, 0],
#         }
#     )
#     intermediate = PandasEdgeMap(df, weight_label="weight")
#     y = dpr.translate(x, PandasEdgeMap)
#     dpr.assert_equal(y, intermediate)
#     # Convert networkx <- pandas edge list
#     x2 = dpr.translate(y, NetworkXGraph)
#     dpr.assert_equal(x, x2)
