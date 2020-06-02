from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx


if has_pandas and has_networkx:
    from .types import PandasEdgeMap
    import networkx as nx
    from ..networkx.types import NetworkXEdgeMap

    @translator
    def edgemap_from_networkx(x: NetworkXEdgeMap, **props) -> PandasEdgeMap:
        df = nx.convert_matrix.to_pandas_edgelist(
            x.value, source="source", target="destination"
        )
        cols = ["source", "destination", x.weight_label]
        df = df[cols]
        return PandasEdgeMap(
            df,
            src_label="source",
            dst_label="destination",
            weight_label=x.weight_label,
            is_directed=x.value.is_directed(),
        )
