from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx, has_scipy
import operator

if has_pandas:
    from .types import PandasEdgeMap, PandasEdgeSet

    @translator
    def edgemap_to_edgeset(x: PandasEdgeMap, **props) -> PandasEdgeSet:
        return PandasEdgeSet(x.value, x.src_label, x.dst_label)


if has_pandas and has_scipy:
    import pandas as pd
    import scipy.sparse as ss
    from ..scipy.types import ScipyEdgeMap, ScipyEdgeSet

    @translator
    def scipy_edgemap_to_pandas_edgemap(x: ScipyEdgeMap, **props) -> PandasEdgeMap:
        is_directed = ScipyEdgeMap.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rcw_triples = zip(row_ids, column_ids, coo_matrix.data)
        if not is_directed:
            rcw_triples = filter(lambda triple: triple[0] < triple[1], rcw_triples)
        df = pd.DataFrame(rcw_triples, columns=["source", "target", "weight"])
        return PandasEdgeMap(df, is_directed=is_directed)

    @translator
    def pandas_edgemap_to_scipy_edgemap(x: PandasEdgeMap, **props) -> ScipyEdgeMap:
        is_directed = x.is_directed
        node_list = pd.unique(x.value[[x.src_label, x.dst_label]].values.ravel("K"))
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        source_positions = x.value[x.src_label].map(get_id_pos)
        target_positions = x.value[x.dst_label].map(get_id_pos)
        weights = x.value[x.weight_label]
        if not is_directed:
            source_positions, target_positions = (
                pd.concat([source_positions, target_positions]),
                pd.concat([target_positions, source_positions]),
            )
            weights = pd.concat([weights, weights])
        matrix = ss.coo_matrix(
            (weights, (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeMap(matrix, node_list)

    @translator
    def scipy_edgeset_to_pandas_edgeset(x: ScipyEdgeSet, **props) -> PandasEdgeSet:
        is_directed = ScipyEdgeSet.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rc_pairs = zip(row_ids, column_ids)
        if not is_directed:
            rc_pairs = filter(lambda pair: pair[0] < pair[1], rc_pairs)
        df = pd.DataFrame(rc_pairs, columns=["source", "target"])
        return PandasEdgeSet(df, is_directed=is_directed)

    @translator
    def pandas_edgeset_to_scipy_edgeset(x: PandasEdgeSet, **props) -> ScipyEdgeSet:
        is_directed = x.is_directed
        node_list = pd.unique(x.value[[x.src_label, x.dst_label]].values.ravel("K"))
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        source_positions = x.value[x.src_label].map(get_id_pos)
        target_positions = x.value[x.dst_label].map(get_id_pos)
        if not is_directed:
            source_positions, target_positions = (
                pd.concat([source_positions, target_positions]),
                pd.concat([target_positions, source_positions]),
            )
        matrix = ss.coo_matrix(
            (np.ones(len(source_positions)), (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeSet(matrix, node_list)


if has_pandas and has_networkx:
    from .types import PandasEdgeMap
    import networkx as nx
    from ..networkx.types import NetworkXGraph

    # @translator
    # def edgemap_from_networkx(x: NetworkXGraph, **props) -> PandasEdgeMap:
    #     df = nx.convert_matrix.to_pandas_edgelist(
    #         x.value, source="source", target="destination"
    #     )
    #     cols = ["source", "destination", x.weight_label]
    #     df = df[cols]
    #     return PandasEdgeMap(
    #         df,
    #         src_label="source",
    #         dst_label="destination",
    #         weight_label=x.weight_label,
    #         is_directed=x.value.is_directed(),
    #     )
