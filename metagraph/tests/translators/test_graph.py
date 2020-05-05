import pytest
from metagraph.tests.util import default_plugin_resolver
from metagraph.plugins.scipy.types import ScipyAdjacencyMatrix
from metagraph.plugins.networkx.types import NetworkXGraph
from metagraph.plugins.graphblas.types import GrblasAdjacencyMatrix
from metagraph.plugins.pandas.types import PandasEdgeList
from metagraph import IndexedNodes
import networkx as nx
import scipy.sparse as ss
import grblas
import pandas as pd
import numpy as np


def test_networkx_scipy(default_plugin_resolver):
    dpr = default_plugin_resolver
    nidx = IndexedNodes("CAB")
    g = nx.DiGraph()
    g.add_weighted_edges_from(
        [("A", "A", 1), ("A", "B", 2), ("B", "B", 0), ("B", "C", 3), ("C", "B", 3)]
    )
    x = NetworkXGraph(g, weight_label="weight", node_index=nidx)
    # Convert networkx -> scipy adjacency
    #    C A B
    # C [    3]
    # A [  1 2]
    # B [3   0]
    m = ss.coo_matrix(
        ([3, 1, 2, 3, 0], ([0, 1, 1, 2, 2], [2, 1, 2, 0, 2])), dtype=np.int64
    )
    intermediate = ScipyAdjacencyMatrix(m, node_index=nidx)
    y = dpr.translate(x, ScipyAdjacencyMatrix)
    ScipyAdjacencyMatrix.Type.assert_equal(y, intermediate)


def test_scipy_graphblas(default_plugin_resolver):
    dpr = default_plugin_resolver
    nidx = IndexedNodes("ABC")
    #    A B C
    # A [1 2  ]
    # B [  0 3]
    # C [  3  ]
    g = ss.coo_matrix(
        ([1, 2, 0, 3, 3], ([0, 0, 1, 1, 2], [0, 1, 1, 2, 1])), dtype=np.int64
    )
    x = ScipyAdjacencyMatrix(g, node_index=nidx)
    # Convert scipy adjacency to graphblas
    m = grblas.Matrix.from_values(
        [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3], dtype=grblas.dtypes.INT64
    )
    intermediate = GrblasAdjacencyMatrix(m, node_index=nidx)
    y = dpr.translate(x, GrblasAdjacencyMatrix)
    GrblasAdjacencyMatrix.Type.assert_equal(y, intermediate)


def test_networkx_2_pandas(default_plugin_resolver):
    dpr = default_plugin_resolver
    nidx = IndexedNodes("CAB")
    g = nx.DiGraph()
    g.add_weighted_edges_from(
        [("A", "A", 1), ("A", "B", 2), ("B", "B", 0), ("B", "C", 3), ("C", "B", 3)]
    )
    x = NetworkXGraph(g, weight_label="weight", node_index=nidx)
    # Convert networkx -> pandas edge list
    df = pd.DataFrame(
        {
            "source": ["A", "A", "B", "C", "B"],
            "target": ["A", "B", "C", "B", "B"],
            "weight": [1, 2, 3, 3, 0],
        }
    )
    intermediate = PandasEdgeList(df, weight_label="weight", node_index=nidx)
    y = dpr.translate(x, PandasEdgeList)
    PandasEdgeList.Type.assert_equal(y, intermediate)
    # Convert networkx <- pandas edge list
    x2 = dpr.translate(y, NetworkXGraph)
    NetworkXGraph.Type.assert_equal(x, x2)
