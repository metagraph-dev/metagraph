from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx

if has_pandas:
    import pandas as pd
    from .types import PandasEdgeList

if has_pandas and has_networkx:
    import networkx as nx
    from ..networkx.types import NetworkXGraphType

    @translator
    def graph_from_networkx(x: NetworkXGraphType, **props) -> PandasEdgeList:
        df = nx.convert_matrix.to_pandas_edgelist(
            x, source="source", target="destination"
        )
        return PandasEdgeList(df, src_label="source", dest_label="destination")
