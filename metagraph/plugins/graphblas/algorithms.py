from metagraph import concrete_algorithm, NodeID
from metagraph.plugins import has_grblas
from typing import Tuple, Iterable, Any, Union, Optional
import random
import numpy as np

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
        node_list, _ = graph.nodes.to_values()
        node_list = list(node_list)
        A = graph.value[node_list, node_list].new()
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
        A = graph.value
        N = A.ncols
        scale_edges = A.apply(gb.unary.one).new(dtype=float)
        node_scale = scale_edges.reduce_rows().new()  # num edges
        node_scale << node_scale.apply(gb.unary.minv)  # 1 / num_edges
        index, vals = node_scale.to_values()  # TODO: implement diag and use here
        node_scale_diag = gb.Matrix.from_values(index, index, vals, ncols=N, nrows=N)
        scale_edges(mask=scale_edges.S)[:, :] = damping
        scale_edges << scale_edges.T.mxm(node_scale_diag)  # 0.85 / num_edges

        # `base` vector gets added to the result every iteration
        base = gb.Vector.new(float, N)
        base[:] = (1 - damping) / N

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
        aprops = {
            "edge_type": "map" if isinstance(edges, GrblasEdgeMap) else "set",
            "node_type": "map" if isinstance(nodes, GrblasNodeMap) else "set",
        }
        m = edges.value
        if nodes is not None:
            nodes = nodes.value
            size = nodes.size
            if m.nrows < size:
                resized = gb.Matrix.new(m.dtype, nrows=size, ncols=size)
                resized[: m.nrows, : m.nrows] << m
                m = resized
        return GrblasGraph(m, nodes=nodes, aprops=aprops)

    @concrete_algorithm("subgraph.extract_subgraph")
    def grblas_extract_subgraph(
        graph: GrblasGraph, nodes: GrblasNodeSet
    ) -> GrblasGraph:
        aprops = GrblasGraph.Type.compute_abstract_properties(
            graph, {"is_directed", "node_type", "edge_type", "node_dtype", "edge_dtype"}
        )
        g = graph.value
        chosen_nodes, _ = nodes.value.to_values()
        chosen_nodes = list(chosen_nodes)
        g2 = gb.Matrix.new(g.dtype, g.nrows, g.ncols)
        g2[chosen_nodes, chosen_nodes] << g[chosen_nodes, chosen_nodes].new()

        n = graph.nodes
        n2 = gb.Vector.new(n.dtype, n.size)
        n2[chosen_nodes] << n[chosen_nodes].new()
        gg = GrblasGraph(g2, n2)
        GrblasGraph.Type.preset_abstract_properties(gg, **aprops)
        return gg

    @concrete_algorithm("subgraph.sample.node_sampling")
    def grblas_node_sampling(graph: GrblasGraph, p: float) -> GrblasGraph:
        rand = random.random
        all_nodes, _ = graph.nodes.to_values()
        chosen_nodes = [n for n in all_nodes if rand() < p]
        chosen_nodes = gb.Vector.from_values(chosen_nodes, [1] * len(chosen_nodes))
        return grblas_extract_subgraph(graph, GrblasNodeSet(chosen_nodes))

    @concrete_algorithm("subgraph.sample.edge_sampling")
    def grblas_edge_sampling(graph: GrblasGraph, p: float) -> GrblasGraph:
        aprops = GrblasGraph.Type.compute_abstract_properties(graph, "node_type")
        rand = random.random
        rows, cols, vals = graph.value.to_values()
        chosen_edges = np.array([i for i in range(len(vals)) if rand() < p])
        # TODO: fix this once `to_values()` returns ndarray
        rows = np.array(rows)[chosen_edges]
        cols = np.array(cols)[chosen_edges]
        vals = np.array(vals)[chosen_edges]
        chosen_nodes = np.intersect1d(rows, cols)
        m = gb.Matrix.from_values(rows, cols, vals)
        if aprops["node_type"] == "map":
            nidx, nvals = graph.nodes.to_values()
            nidx = np.array(nidx)
            nvals = np.array(nvals)[nidx.searchsorted(chosen_nodes)]
            nodes = gb.Vector.from_values(chosen_nodes, nvals)
        else:
            nodes = gb.Vector.from_values(chosen_nodes, [1] * len(chosen_nodes))
        return GrblasGraph(m, nodes)

    @concrete_algorithm("subgraph.sample.ties")
    def grblas_totally_induced_edge_sampling(
        graph: GrblasGraph, p: float
    ) -> GrblasGraph:
        rand = random.random
        rows, cols, vals = graph.value.to_values()
        chosen_edges = np.array([i for i in range(len(vals)) if rand() < p])
        # TODO: fix this once `to_values()` returns ndarray
        rows = np.array(rows)[chosen_edges]
        cols = np.array(cols)[chosen_edges]
        chosen_nodes = np.intersect1d(rows, cols)
        chosen_nodes = gb.Vector.from_values(chosen_nodes, [1] * len(chosen_nodes))
        return grblas_extract_subgraph(graph, GrblasNodeSet(chosen_nodes))
