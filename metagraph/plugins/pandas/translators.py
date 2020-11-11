from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx, has_scipy

if has_pandas:
    from .types import PandasEdgeMap, PandasEdgeSet

    @translator
    def edgemap_to_edgeset(x: PandasEdgeMap, **props) -> PandasEdgeSet:
        return PandasEdgeSet(
            x.value, x.src_label, x.dst_label, is_directed=x.is_directed
        )


if has_pandas and has_scipy:
    import pandas as pd
    from ..scipy.types import ScipyEdgeMap, ScipyEdgeSet

    @translator
    def edgemap_from_scipy(x: ScipyEdgeMap, **props) -> PandasEdgeMap:
        is_directed = ScipyEdgeMap.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rcw_triples = zip(row_ids, column_ids, coo_matrix.data)
        if not is_directed:
            rcw_triples = filter(lambda triple: triple[0] <= triple[1], rcw_triples)
        df = pd.DataFrame(rcw_triples, columns=["source", "target", "weight"])
        return PandasEdgeMap(df, is_directed=is_directed)

    @translator
    def edgeset_from_scipy(x: ScipyEdgeSet, **props) -> PandasEdgeSet:
        is_directed = ScipyEdgeSet.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rc_pairs = zip(row_ids, column_ids)
        if not is_directed:
            rc_pairs = filter(lambda pair: pair[0] <= pair[1], rc_pairs)
        df = pd.DataFrame(rc_pairs, columns=["source", "target"])
        return PandasEdgeSet(df, is_directed=is_directed)
