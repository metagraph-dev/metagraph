from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx

if has_networkx:
    import networkx as nx
    from .types import NetworkXGraph, AutoNetworkXDiGraphType, AutoNetworkXGraphType

    @translator
    def nxundirected_from_nxgraph(x: NetworkXGraph, **props) -> AutoNetworkXGraphType:
        # TODO: update all graph dicts if weight_label != "weight" ???
        # TODO: what should the behavior be if x is directed?
        return x.value

    @translator
    def nxdirected_from_nxgraph(x: NetworkXGraph, **props) -> AutoNetworkXDiGraphType:
        # TODO: update all graph dicts if weight_label != "weight" ???
        if x.value.is_directed():
            return x.value
        else:
            return x.value.to_directed()

    @translator
    def nxgraph_from_nxundirected(x: AutoNetworkXGraphType, **props) -> NetworkXGraph:
        type_info = AutoNetworkXGraphType.get_type(x)
        weights = type_info["weights"]
        if weights == "unweighted":
            return NetworkXGraph(x)
        else:
            dtype = type_info["dtype"]
            return NetworkXGraph(x, weight_label="weight", weights=weights, dtype=dtype)

    @translator
    def nxgraph_from_nxdirected(x: AutoNetworkXDiGraphType, **props) -> NetworkXGraph:
        type_info = AutoNetworkXDiGraphType.get_type(x)
        weights = type_info["weights"]
        if weights == "unweighted":
            return NetworkXGraph(x)
        else:
            dtype = type_info["dtype"]
            return NetworkXGraph(x, weight_label="weight", weights=weights, dtype=dtype)


if has_networkx and has_pandas:
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
