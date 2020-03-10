from metagraph import concrete_algorithm
from metagraph.plugins import has_scipy
from .types import ScipyAdjacencyMatrix


if has_scipy:
    import scipy.sparse as ss

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
