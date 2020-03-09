from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx

if has_networkx:
    import networkx as nx
    from .wrappers import NetworkXGraphType


if has_networkx and has_pandas:
    from ..pandas.wrappers import PandasEdgeList

    @translator
    def graph_from_pandas(x: PandasEdgeList, **props) -> NetworkXGraphType:
        g = x.value[[x.src_label, x.dest_label]]
        out = nx.DiGraph()
        out.add_edges_from(g.itertuples(index=False, name="Edge"))
        return out
