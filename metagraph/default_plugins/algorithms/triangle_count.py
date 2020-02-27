from ... import abstract_algorithm, concrete_algorithm
from ..abstract_types import Graph
from ..wrappers.scipy import ScipyAdjacencyMatrix
from .. import networkx, scipy


@abstract_algorithm("cluster.triangle_count")
def triangle_count(graph: Graph) -> int:
    """
    Counts the number of unique triangles in an undirected graph
    """
    pass


if networkx:
    nx = networkx

    @concrete_algorithm("cluster.triangle_count")
    def nx_triangle_count(graph: nx.DiGraph) -> int:
        # NetworkX's algorithm only works on undirected graphs
        ugraph = graph.to_undirected()
        triangles = nx.triangles(ugraph)
        # Sum up triangles from each node
        # Divide by 3 becuase each triangle is counted 3 times
        total_triangles = sum(triangles.values()) // 3
        return total_triangles


if scipy:
    ss = scipy.sparse

    @concrete_algorithm("cluster.triangle_count")
    def ss_triangle_count(graph: ScipyAdjacencyMatrix) -> int:
        """
        Uses the triangle counting method descripbed in
        https://www.sandia.gov/~srajama/publications/Tricount-HPEC.pdf
        """
        L = ss.tril(graph.value, k=-1).tocsr()
        U = ss.triu(graph.value, k=1).tocsc()
        # Check for inequality, which indicates non-symmetry
        if (L != U.T).nnz > 0:
            raise ValueError("adjacency matrix must be symmetric for triangle count")
        # https://www.sandia.gov/~srajama/publications/Tricount-HPEC.pdf
        return int((L @ U.T).multiply(L).sum())
