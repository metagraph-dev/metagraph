from metagraph import translator
from metagraph.plugins import has_networkx, has_scipy


if has_networkx and has_scipy:
    import networkx as nx
    import numpy as np
    from .types import NetworkXGraph
    from ..scipy.types import ScipyGraph

    @translator
    def graph_from_scipy(x: ScipyGraph, **props) -> NetworkXGraph:
        from ..python.types import dtype_casting

        aprops = ScipyGraph.Type.compute_abstract_properties(
            x, {"is_directed", "edge_type", "edge_dtype", "node_type", "node_dtype"}
        )

        nx_graph = nx.from_scipy_sparse_matrix(
            x.value,
            create_using=nx.DiGraph if aprops["is_directed"] else nx.Graph,
            edge_attribute="weight",
        )

        if aprops["edge_type"] == "set":
            # Remove weight attribute
            for _, _, attr in nx_graph.edges(data=True):
                del attr["weight"]
        else:
            caster = dtype_casting[aprops["edge_dtype"]]
            for _, _, attr in nx_graph.edges(data=True):
                attr["weight"] = caster(attr["weight"])

        is_sequential_node_list = (x.node_list == np.arange(len(x.node_list))).all()
        if not is_sequential_node_list:
            pos2id = dict(enumerate(x.node_list))
            nx.relabel_nodes(nx_graph, pos2id, False)

        if x.node_vals is not None:
            caster = dtype_casting[aprops["node_dtype"]]
            node_weights = {
                idx: caster(val) for idx, val in zip(x.node_list, x.node_vals)
            }
            nx.set_node_attributes(nx_graph, node_weights, name="weight")

        return NetworkXGraph(nx_graph, aprops=aprops)
