import pytest
from metagraph.plugins.scipy.types import ScipyAdjacencyMatrix
from metagraph.plugins.networkx.types import NetworkXGraph
from metagraph.plugins.pandas.types import PandasEdgeList
from metagraph.plugins.graphblas.types import GrblasAdjacencyMatrix
from metagraph import IndexedNodes
import scipy.sparse as ss
import networkx as nx
import pandas as pd
import grblas


def test_scipy_adj():
    pytest.xfail("not written")


def test_networkx():
    pytest.xfail("not written")


def test_pandas_edge():
    pytest.xfail("not written")


def test_graphblas_adj():
    # [1 2  ]
    # [  0 3]
    # [  3  ]
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            )
        ),
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            )
        ),
    )
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1.0000000000001, 2, 0, 3, 3]
            )
        ),
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1.0, 2, 0, 3, 3]
            )
        ),
    )
    assert not GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            )
        ),
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 3, 0, 3, 3]
            )
        ),
    )
    assert not GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            )
        ),
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 0], [1, 2, 0, 3, 3]
            )
        ),
    )
    assert not GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            )
        ),
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 1, 2], [1, 2, 0, 3, 3, 0]
            )
        ),
    )
    # weights don't match, so we take the fast path and declare them not equal
    assert not GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            )
        ),
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            ),
            weights="any",
        ),
    )
    # Node index affects comparison
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            ),
            node_index=IndexedNodes("ABC"),
        ),
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 2, 2], [0, 1, 0, 0, 2], [0, 3, 3, 2, 1]
            ),
            node_index=IndexedNodes("BCA"),
        ),
    )
    # Transposed
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            )
        ),
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 1, 1, 1, 2], [0, 0, 1, 2, 1], [1, 2, 0, 3, 3]
            ),
            transposed=True,
        ),
    )
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            ),
            transposed=True,
        ),
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 1, 1, 1, 2], [0, 0, 1, 2, 1], [1, 2, 0, 3, 3]
            )
        ),
    )
    assert GrblasAdjacencyMatrix.Type.compare_objects(
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            ),
            transposed=True,
        ),
        GrblasAdjacencyMatrix(
            grblas.Matrix.new_from_values(
                [0, 0, 1, 1, 2], [0, 1, 1, 2, 1], [1, 2, 0, 3, 3]
            ),
            transposed=True,
        ),
    )
    with pytest.raises(TypeError):
        GrblasAdjacencyMatrix.Type.compare_objects(5, 5)
