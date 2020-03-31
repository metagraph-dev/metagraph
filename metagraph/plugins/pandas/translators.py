from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx


if has_pandas and has_networkx:
    from .types import PandasEdgeList
    import networkx as nx
    from ..networkx.types import NetworkXGraph

    @translator
    def graph_from_networkx(x: NetworkXGraph, **props) -> PandasEdgeList:
        type_info = NetworkXGraph.Type.get_type(x)
        df = nx.convert_matrix.to_pandas_edgelist(
            x.value, source="source", target="destination"
        )
        cols = ["source", "destination"]
        if x.weight_label:
            cols.append(x.weight_label)
        df = df[cols]
        return PandasEdgeList(
            df,
            src_label="source",
            dst_label="destination",
            weight_label=x.weight_label,
            is_directed=type_info["is_directed"],
            weights=type_info["weights"],
            node_index=x.node_index,
        )
