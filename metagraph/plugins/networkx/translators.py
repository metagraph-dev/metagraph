from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx


if has_networkx and has_pandas:
    import networkx as nx
    from .types import NetworkXGraph
    from ..pandas.types import PandasEdgeList

    @translator
    def graph_from_pandas(x: PandasEdgeList, **props) -> NetworkXGraph:
        if x.is_directed:
            out = nx.DiGraph()
        else:
            out = nx.Graph()

        if x.weight_label is None:
            g = x.value[[x.src_label, x.dst_label]]
            out.add_edges_from(g.itertuples(index=False, name="Edge"))
            return NetworkXGraph(out)
        else:
            type_info = PandasEdgeList.Type.get_type(x)
            g = x.value[[x.src_label, x.dst_label, x.weight_label]]
            out.add_weighted_edges_from(g.itertuples(index=False, name="WeightedEdge"))
            return NetworkXGraph(
                out,
                weight_label="weight",
                weights=type_info.prop_val["weights"],
                dtype=type_info.prop_val["dtype"],
                node_index=x.node_index,
            )
