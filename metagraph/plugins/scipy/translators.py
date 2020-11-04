from metagraph import translator
from metagraph.plugins import has_scipy, has_networkx, has_grblas, has_pandas
from metagraph.plugins.numpy.types import NumpyNodeSet, NumpyNodeMap
import numpy as np

if has_scipy:
    import scipy.sparse as ss
    from .types import ScipyEdgeMap, ScipyEdgeSet, ScipyMatrixType
    from ..numpy.types import NumpyMatrix

    @translator
    def edgemap_to_edgeset(x: ScipyEdgeMap, **props) -> ScipyEdgeSet:
        aprops = ScipyEdgeMap.Type.compute_abstract_properties(x, {"is_directed"})
        data = x.value.copy()
        # Force all values to be 1's to indicate no weights
        data.data = np.ones_like(data.data)
        ses = ScipyEdgeSet(data, x.node_list)
        ScipyEdgeSet.Type.preset_abstract_properties(ses, **aprops)
        return ses

    @translator
    def matrix_from_numpy(x: NumpyMatrix, **props) -> ScipyMatrixType:
        # scipy.sparse assumes zero is empty
        # To work around this limitation, we use a mask
        # and directly set `.data` after construction
        if x.mask is None:
            if (x.value == 0).any():
                mask = x.value.copy()
                mask[:, :] = 1
                mat = ss.coo_matrix(mask)
                mat.data = x.value.flatten()
            else:
                mat = ss.coo_matrix(x.value)
        else:
            mat = ss.coo_matrix(x.mask)
            mat.data = x.value[x.mask]
        return mat


if has_scipy and has_networkx:
    import networkx as nx
    from .types import ScipyGraph
    from ..networkx.types import NetworkXGraph

    @translator
    def graph_from_networkx(x: NetworkXGraph, **props) -> ScipyGraph:
        aprops = NetworkXGraph.Type.compute_abstract_properties(
            x, {"node_type", "edge_type", "node_dtype", "edge_dtype", "is_directed"}
        )
        node_list = list(sorted(x.value.nodes()))
        node_vals = None
        if aprops["node_type"] == "map":
            node_vals = np.array(
                [x.value.nodes[n].get(x.node_weight_label) for n in node_list]
            )

        if aprops["edge_type"] == "map":
            m = nx.convert_matrix.to_scipy_sparse_matrix(
                x.value, nodelist=node_list, weight=x.edge_weight_label,
            )
        else:
            m = nx.convert_matrix.to_scipy_sparse_matrix(x.value, nodelist=node_list)

        sg = ScipyGraph(m, node_list, node_vals)
        ScipyGraph.Type.preset_abstract_properties(sg, **aprops)
        return sg


if has_scipy and has_grblas:
    import scipy.sparse as ss
    from .types import ScipyMatrixType
    from ..graphblas.types import (
        GrblasMatrixType,
        GrblasGraph,
        GrblasEdgeSet,
        GrblasEdgeMap,
        dtype_grblas_to_mg,
        find_active_nodes,
    )

    @translator
    def matrix_from_graphblas(x: GrblasMatrixType, **props) -> ScipyMatrixType:
        rows, cols, vals = x.to_values()
        mat = ss.coo_matrix((vals, (rows, cols)), x.shape)
        return mat

    @translator
    def edgeset_from_graphblas(x: GrblasEdgeSet, **props) -> ScipyEdgeSet:
        aprops = GrblasEdgeSet.Type.compute_abstract_properties(x, {"is_directed"})
        active_nodes = find_active_nodes(x.value)
        gm = x.value[active_nodes, active_nodes].new()
        rows, cols, _ = gm.to_values()
        sm = ss.coo_matrix(
            ([True] * len(rows), (rows, cols)), shape=gm.shape, dtype=bool
        )
        ses = ScipyEdgeSet(sm, node_list=active_nodes)
        ScipyEdgeSet.Type.preset_abstract_properties(ses, **aprops)
        return ses

    @translator
    def edgemap_from_graphblas(x: GrblasEdgeMap, **props) -> ScipyEdgeMap:
        aprops = GrblasEdgeMap.Type.compute_abstract_properties(x, {"is_directed"})
        active_nodes = find_active_nodes(x.value)
        gm = x.value[active_nodes, active_nodes].new()
        rows, cols, vals = gm.to_values()
        sm = ss.coo_matrix(
            (vals, (rows, cols)),
            dtype=dtype_grblas_to_mg[x.value.dtype.name],
            shape=gm.shape,
        )
        sem = ScipyEdgeMap(sm, node_list=active_nodes)
        ScipyEdgeMap.Type.preset_abstract_properties(sem, **aprops)
        return sem

    @translator(include_resolver=True)
    def graph_from_graphblas(x: GrblasGraph, *, resolver, **props) -> ScipyGraph:
        aprops = GrblasGraph.Type.compute_abstract_properties(
            x, {"node_type", "edge_type", "node_dtype", "edge_dtype", "is_directed"}
        )

        node_list, node_vals = x.nodes.to_values()
        node_list = np.array(node_list)
        if aprops["node_type"] == "map":
            node_vals = np.array(node_vals)
        else:
            node_vals = None
        size = len(node_list)

        compressed = x.value[node_list, node_list].new()
        rows, cols, vals = compressed.to_values()

        if aprops["edge_type"] == "map":
            dtype = dtype_grblas_to_mg[x.value.dtype.name]
            matrix = ss.coo_matrix(
                (vals, (rows, cols)), shape=(size, size), dtype=dtype
            )
        elif aprops["edge_type"] == "set":
            ones = np.ones_like(rows)
            matrix = ss.coo_matrix((ones, (rows, cols)), shape=(size, size), dtype=bool)
        else:
            raise TypeError(f"Cannot translate with edge_type={aprops['edge_type']}")

        sg = ScipyGraph(matrix, node_list, node_vals)
        ScipyGraph.Type.preset_abstract_properties(sg, **aprops)
        return sg


if has_scipy and has_pandas:
    import pandas as pd
    from ..pandas.types import PandasEdgeMap, PandasEdgeSet

    @translator
    def edgemap_from_pandas(x: PandasEdgeMap, **props) -> ScipyEdgeMap:
        is_directed = x.is_directed
        node_list = pd.unique(x.value[[x.src_label, x.dst_label]].values.ravel("K"))
        node_list.sort()
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        source_positions = x.value[x.src_label].map(get_id_pos)
        target_positions = x.value[x.dst_label].map(get_id_pos)
        weights = x.value[x.weight_label]
        if not is_directed:
            nonself = source_positions != target_positions
            source_positions, target_positions = (
                pd.concat([source_positions, target_positions[nonself]]),
                pd.concat([target_positions, source_positions[nonself]]),
            )
            weights = pd.concat([weights, weights[nonself]])
        matrix = ss.coo_matrix(
            (weights, (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        ss_edgemap = ScipyEdgeMap(matrix, node_list)
        ScipyEdgeMap.Type.preset_abstract_properties(
            ss_edgemap, is_directed=is_directed
        )
        return ss_edgemap

    @translator
    def edgeset_from_pandas(x: PandasEdgeSet, **props) -> ScipyEdgeSet:
        is_directed = x.is_directed
        node_list = pd.unique(x.value[[x.src_label, x.dst_label]].values.ravel("K"))
        node_list.sort()
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        source_positions = x.value[x.src_label].map(get_id_pos)
        target_positions = x.value[x.dst_label].map(get_id_pos)
        if not is_directed:
            nonself = source_positions != target_positions
            source_positions, target_positions = (
                pd.concat([source_positions, target_positions[nonself]]),
                pd.concat([target_positions, source_positions[nonself]]),
            )
        matrix = ss.coo_matrix(
            (np.ones(len(source_positions)), (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        ss_edgeset = ScipyEdgeSet(matrix, node_list)
        # Set is_directed property
        ScipyEdgeSet.Type.preset_abstract_properties(
            ss_edgeset, is_directed=is_directed
        )
        return ss_edgeset
