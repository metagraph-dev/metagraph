from metagraph import concrete_algorithm
from metagraph.plugins import has_networkx


if has_networkx:
    import networkx as nx
    from ..python.wrappers import PythonSparseVector

    @concrete_algorithm("link_analysis.pagerank")
    def pagerank(
        graph: nx.DiGraph,
        damping: float = 0.85,
        maxiter: int = 50,
        tolerance: float = 1e-05,
    ) -> PythonSparseVector:
        pagerank = nx.pagerank(
            graph, alpha=damping, max_iter=maxiter, tol=tolerance, weight=None
        )
        return PythonSparseVector(pagerank)

    @concrete_algorithm("cluster.triangle_count")
    def triangle_count(graph: nx.DiGraph) -> int:
        # NetworkX's algorithm only works on undirected graphs
        ugraph = graph.to_undirected()
        triangles = nx.triangles(ugraph)
        # Sum up triangles from each node
        # Divide by 3 becuase each triangle is counted 3 times
        total_triangles = sum(triangles.values()) // 3
        return total_triangles
