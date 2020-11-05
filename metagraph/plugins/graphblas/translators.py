import numpy as np
from metagraph import translator
from metagraph.plugins import has_grblas, has_scipy
from ..numpy.types import NumpyVector, NumpyNodeMap, NumpyNodeSet
from ..python.types import PythonNodeSet


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
    def vector_from_numpy(x: NumpyVector, **props) -> GrblasVectorType:
        idx = np.arange(len(x))
        if x.mask is not None:
            idx = idx[x.mask]
        vals = x.value[idx]
        vec = grblas.Vector.from_values(
            idx, vals, size=len(x), dtype=dtype_mg_to_grblas[x.value.dtype]
        )
        return vec

    @translator
    def nodeset_from_python(x: PythonNodeSet, **props) -> GrblasNodeSet:
        nodes = list(sorted(x.value))
        size = nodes[-1] + 1
        vec = grblas.Vector.from_values(nodes, [1] * len(nodes), size=size)
        return GrblasNodeSet(vec)

    @translator
    def nodeset_from_numpy(x: NumpyNodeSet, **props) -> GrblasNodeSet:
        if x.mask is not None:
            idx = np.flatnonzero(x.mask)
        else:
            idx = x.node_array
        size = idx[-1] + 1
        vec = grblas.Vector.from_values(idx, [True] * len(idx), size=size, dtype=bool)
        return GrblasNodeSet(vec)

    @translator
    def nodemap_from_numpy(x: NumpyNodeMap, **props) -> GrblasNodeMap:
        if x.mask is not None:
            idx = np.flatnonzero(x.mask)
            vals = x.value[idx]
        elif x.id2pos is not None:
            idx = x.pos2id
            vals = x.value
        else:
            idx = np.arange(len(x.value))
            vals = x.value
        size = idx[-1] + 1
        vec = grblas.Vector.from_values(
            idx, vals, size=size, dtype=dtype_mg_to_grblas[x.value.dtype]
        )
        return GrblasNodeMap(vec)


if has_grblas and has_scipy:
    from ..scipy.types import ScipyEdgeSet, ScipyEdgeMap, ScipyGraph, ScipyMatrixType
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

    @translator
    def matrix_from_scipy(x: ScipyMatrixType, **props) -> GrblasMatrixType:
        x = x.tocoo()
        nrows, ncols = x.shape
        dtype = dtype_mg_to_grblas[x.dtype]
        vec = grblas.Matrix.from_values(
            x.row, x.col, x.data, nrows=nrows, ncols=ncols, dtype=dtype
        )
        return vec
