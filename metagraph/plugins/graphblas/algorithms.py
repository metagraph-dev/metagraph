from metagraph import concrete_algorithm, NodeID
from metagraph.plugins import has_grblas
from typing import Tuple, Iterable, Any, Union, Optional
import random

if has_grblas:
    import grblas as gb
    from .types import (
        GrblasEdgeSet,
        GrblasEdgeMap,
        GrblasGraph,
        GrblasNodeMap,
        GrblasNodeSet,
        GrblasVectorType,
    )

    @concrete_algorithm("cluster.triangle_count")
    def grblas_triangle_count(graph: GrblasGraph) -> int:
        # Burkhardt method: num_triangles = sum(sum(A @ A) * A) / 6
        # We do it in two steps: a matrix multiplication then a reduction
        A = graph.edges.value
        val = A.mxm(
            A.T,  # Transpose here assumes symmetric matrix stored by row (the default for SuiteSparse:GraphBLAS)
            gb.semiring.plus_pair[
                gb.dtypes.UINT64
            ],  # `pair` binary operator returns 1; be dtype-agnostic
        ).new(
            mask=A.S
        )  # Using a (structural) mask is equivalent to the elementwise multiplication step
        return val.reduce_scalar().value // 6

    @concrete_algorithm("centrality.pagerank")
    def grblas_pagerank(
        graph: GrblasGraph, damping: float, maxiter: int, tolerance: float
    ) -> GrblasNodeMap:
        # `scale_edges` matrix does the bulk of the work; it's what distributes
        # the current value of a vertex to its neighbors
        A = graph.edges.value
        N = A.ncols
        scale_edges = A.apply(gb.unary.one).new(dtype=float)
        node_scale = scale_edges.reduce_rows().new()  # num edges
        node_scale << node_scale.apply(gb.unary.minv)  # 1 / num_edges
        index, vals = node_scale.to_values()  # TODO: implement diag and use here
        node_scale_diag = gb.Matrix.from_values(index, index, vals, ncols=N, nrows=N)
        scale_edges(mask=scale_edges.S)[:, :] = 0.85
        scale_edges << scale_edges.T.mxm(node_scale_diag)  # 0.85 / num_edges

        # `base` vector gets added to the result every iteration
        base = gb.Vector.new(float, N)
        base[:] = 0.15 / N

        # `r` vector holds the results
        r = gb.Vector.new(float, N)
        r[:] = 1 / N

        for i in range(maxiter):
            prev_r = r.dup()
            r << scale_edges.mxv(r)
            r << r.ewise_add(base, gb.monoid.plus)
            # now calculate the difference and check the tolerance
            prev_r << prev_r.ewise_mult(r, gb.binary.minus)
            prev_r << prev_r.apply(gb.unary.abs)
            err = prev_r.reduce().value
            if err < N * tolerance:
                break
        return GrblasNodeMap(r)

    @concrete_algorithm("util.graph.build")
    def grblas_graph_build(
        edges: Union[GrblasEdgeSet, GrblasEdgeMap],
        nodes: Union[GrblasNodeSet, GrblasNodeMap, None],
    ) -> GrblasGraph:
        return GrblasGraph(edges, nodes)

    @concrete_algorithm("subgraph.extract_subgraph")
    def grblas_extract_subgraph(
        graph: GrblasGraph, nodes: GrblasNodeSet
    ) -> GrblasGraph:
        g = graph.edges.value
        chosen_nodes, _ = nodes.value.to_values()
        chosen_nodes = list(chosen_nodes)
        g2 = gb.Matrix.new(g.dtype, g.nrows, g.ncols)
        g2[chosen_nodes, chosen_nodes] << g[chosen_nodes, chosen_nodes].new()
        edges = GrblasEdgeMap(g2)
        # Handle node values
        if graph.nodes is not None and isinstance(graph.nodes, GrblasNodeMap):
            n = graph.nodes.value
            n2 = gb.Vector.new(n.dtype, n.size)
            n2[chosen_nodes] << n2[chosen_nodes].new()
            nodes = n2
        return GrblasGraph(edges, nodes)

    # @concrete_algorithm("subgraph.sample.node_sampling")
    # def grblas_node_sampling(graph: GrblasGraph, p: float) -> GrblasGraph:
    #     if graph.nodes is None:
    #         all_nodes = range(graph.edges.value.nrows)
    #     else:
    #         all_nodes, _ = graph.nodes.value.to_values()
    #     chosen_nodes = [n for n in all_nodes if random.random() < p]
    #     chosen_nodes = gb.Vector.from_values(chosen_nodes, [1]*len(chosen_nodes))
    #     return grblas_extract_subgraph(graph, GrblasNodeSet(chosen_nodes))

    # @concrete_algorithm("subgraph.sample.edge_sampling")
    # def grblas_edge_sampling(graph: GrblasGraph, p: float) -> GrblasGraph:
    #     pass

    # @concrete_algorithm("subgraph.sample.ties")
    # def grblas_totally_induced_edge_sampling(graph: GrblasGraph, p: float) -> GrblasGraph:
    #     pass

    # @concrete_algorithm("subgraph.sample.random_walk")
    # def grblas_random_walk_sampling(
    #     graph: GrblasGraph,
    #     num_steps: Optional[int],
    #     num_nodes: Optional[int],
    #     num_edges: Optional[int],
    #     jump_probability: float,
    #     start_node: Optional[NodeID],
    # ) -> GrblasGraph:
    #     pass
