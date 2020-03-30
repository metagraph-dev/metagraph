from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx


if has_pandas and has_networkx:
    from .types import PandasEdgeList
    import networkx as nx
    from ..networkx.types import NetworkXGraph

    @translator
    def graph_from_networkx(x: NetworkXGraph, **props) -> PandasEdgeList:
        df = nx.convert_matrix.to_pandas_edgelist(
            x.value, source="source", target="destination"
        )
        return PandasEdgeList(df, src_label="source", dst_label="destination")
