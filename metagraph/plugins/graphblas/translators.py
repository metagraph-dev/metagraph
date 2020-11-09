import numpy as np
from metagraph import translator
from metagraph.plugins import has_grblas, has_scipy
from ..numpy.types import NumpyVectorType, NumpyNodeMap, NumpyNodeSet, NumpyMatrixType
from ..python.types import PythonNodeSetType


if has_grblas:
    import grblas
    from .types import (
        GrblasEdgeMap,
        GrblasEdgeSet,
        GrblasGraph,
        GrblasMatrixType,
        GrblasVectorType,
        GrblasNodeSet,
        GrblasNodeMap,
        dtype_mg_to_grblas,
    )

    @translator
    def nodemap_to_nodeset(x: GrblasNodeMap, **props) -> GrblasNodeSet:
        data = x.value.dup()
        # Force all values to be 1's to indicate no weights
        data[:](data.S) << 1
        return GrblasNodeSet(data)

    @translator
    def edgemap_to_edgeset(x: GrblasEdgeMap, **props) -> GrblasEdgeSet:
        aprops = GrblasEdgeMap.Type.compute_abstract_properties(x, "is_directed")
        data = x.value.dup()
        # Force all values to be 1's to indicate no weights
        data[:, :](data.S) << 1
        return GrblasEdgeSet(data, aprops=aprops)

    @translator
    def vector_from_numpy(x: NumpyVectorType, **props) -> GrblasVectorType:
        idx = np.arange(len(x))
        vec = grblas.Vector.from_values(
            idx, x, size=len(x), dtype=dtype_mg_to_grblas[x.dtype]
        )
        return vec

    @translator
    def nodeset_from_python(x: PythonNodeSetType, **props) -> GrblasNodeSet:
        nodes = list(sorted(x))
        size = nodes[-1] + 1
        vec = grblas.Vector.from_values(nodes, [1] * len(nodes), size=size)
        return GrblasNodeSet(vec)

    @translator
    def nodeset_from_numpy(x: NumpyNodeSet, **props) -> GrblasNodeSet:
        idx = x.value
        size = idx[-1] + 1
        vec = grblas.Vector.from_values(idx, [True] * len(idx), size=size, dtype=bool)
        return GrblasNodeSet(vec)

    @translator
    def nodemap_from_numpy(x: NumpyNodeMap, **props) -> GrblasNodeMap:
        size = x.nodes[-1] + 1
        vec = grblas.Vector.from_values(
            x.nodes, x.value, size=size, dtype=dtype_mg_to_grblas[x.value.dtype]
        )
        return GrblasNodeMap(vec)

    @translator
    def matrix_from_numpy(x: NumpyMatrixType, **props) -> GrblasMatrixType:
        nrows, ncols = x.shape
        dtype = dtype_mg_to_grblas[x.dtype]
        rows = (np.arange(nrows * ncols) % nrows).reshape((ncols, nrows)).T.flatten()
        cols = np.arange(nrows * ncols) % ncols
        data = x.flatten()
        vec = grblas.Matrix.from_values(
            rows, cols, data, nrows=nrows, ncols=ncols, dtype=dtype
        )
        return vec


if has_grblas and has_scipy:
    from ..scipy.types import ScipyEdgeSet, ScipyEdgeMap, ScipyGraph
    from .types import dtype_mg_to_grblas

    @translator
    def edgeset_from_scipy(x: ScipyEdgeSet, **props) -> GrblasEdgeSet:
        aprops = ScipyEdgeSet.Type.compute_abstract_properties(x, {"is_directed"})
        m = x.value.tocoo()
        node_list = x.node_list
        size = max(node_list) + 1
        out = grblas.Matrix.from_values(
            node_list[m.row],
            node_list[m.col],
            np.ones_like(m.data),
            nrows=size,
            ncols=size,
        )
        return GrblasEdgeSet(out, aprops=aprops)

    @translator
    def edgemap_from_scipy(x: ScipyEdgeMap, **props) -> GrblasEdgeMap:
        aprops = ScipyEdgeMap.Type.compute_abstract_properties(x, {"is_directed"})
        m = x.value.tocoo()
        node_list = x.node_list
        size = max(node_list) + 1
        dtype = dtype_mg_to_grblas[x.value.dtype]
        out = grblas.Matrix.from_values(
            node_list[m.row],
            node_list[m.col],
            m.data,
            nrows=size,
            ncols=size,
            dtype=dtype,
        )
        return GrblasEdgeMap(out, aprops=aprops)

    @translator
    def graph_from_scipy(x: ScipyGraph, **props) -> GrblasGraph:
        aprops = ScipyGraph.Type.compute_abstract_properties(
            x, {"node_type", "edge_type", "node_dtype", "edge_dtype", "is_directed"}
        )

        size = x.node_list.max() + 1

        if aprops["node_type"] == "map":
            dtype = dtype_mg_to_grblas[x.node_vals.dtype]
            nodes = grblas.Vector.from_values(
                x.node_list, x.node_vals, size=size, dtype=dtype
            )
        elif aprops["node_type"] == "set":
            nodes = grblas.Vector.from_values(
                x.node_list, [True] * len(x.node_list), size=size, dtype=bool
            )
        else:
            raise TypeError(f"Cannot translate with node_type={aprops['node_type']}")

        edges = x.value.tocoo()
        rows = x.node_list[edges.row]
        cols = x.node_list[edges.col]

        if aprops["edge_type"] == "map":
            dtype = dtype_mg_to_grblas[edges.data.dtype]
            matrix = grblas.Matrix.from_values(
                rows, cols, edges.data, nrows=size, ncols=size, dtype=dtype
            )
        elif aprops["edge_type"] == "set":
            matrix = grblas.Matrix.from_values(
                rows, cols, [True] * len(rows), nrows=size, ncols=size, dtype=bool
            )
        else:
            raise TypeError(f"Cannot translate with edge_type={aprops['edge_type']}")

        return GrblasGraph(matrix, nodes=nodes, aprops=aprops)
