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
        row_ids = x.node_list[coo_matrix.row]
        column_ids = x.node_list[coo_matrix.col]
        weights = coo_matrix.data
        if not is_directed:
            mask = row_ids <= column_ids
            row_ids = row_ids[mask]
            column_ids = column_ids[mask]
            weights = weights[mask]
        df = pd.DataFrame({"source": row_ids, "target": column_ids, "weight": weights})
        return PandasEdgeMap(df, is_directed=is_directed)

    @translator
    def edgeset_from_scipy(x: ScipyEdgeSet, **props) -> PandasEdgeSet:
        is_directed = ScipyEdgeSet.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        row_ids = x.node_list[coo_matrix.row]
        column_ids = x.node_list[coo_matrix.col]
        if not is_directed:
            mask = row_ids <= column_ids
            row_ids = row_ids[mask]
            column_ids = column_ids[mask]
        df = pd.DataFrame({"source": row_ids, "target": column_ids})
        return PandasEdgeSet(df, is_directed=is_directed)
