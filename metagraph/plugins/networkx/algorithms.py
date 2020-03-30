from metagraph import concrete_algorithm
from metagraph.plugins import has_networkx


if has_networkx:
    import networkx as nx
    from .types import NetworkXGraph
    from ..python.types import PythonNodes

    @concrete_algorithm("link_analysis.pagerank")
    def nx_pagerank(
        graph: NetworkXGraph,
        damping: float = 0.85,
        maxiter: int = 50,
        tolerance: float = 1e-05,
    ) -> PythonNodes:
        pagerank = nx.pagerank(
            graph.value, alpha=damping, max_iter=maxiter, tol=tolerance, weight=None
        )
        return PythonNodes(pagerank)

    @concrete_algorithm("cluster.triangle_count")
    def nx_triangle_count(graph: NetworkXGraph) -> int:
        triangles = nx.triangles(graph.value)
        # Sum up triangles from each node
        # Divide by 3 because each triangle is counted 3 times
        total_triangles = sum(triangles.values()) // 3
        return total_triangles

    @concrete_algorithm("cluster.triangle_count_by_node")
    def nx_triangle_count_by_node(graph: NetworkXGraph) -> PythonNodes:
        triangles = nx.triangles(graph.value)
        return PythonNodes(triangles)
