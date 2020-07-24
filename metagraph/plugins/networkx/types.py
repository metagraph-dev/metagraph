from typing import Set, Dict, Any
from metagraph.types import Graph
from metagraph.wrappers import GraphWrapper
from metagraph.plugins import has_networkx
import math


if has_networkx:
    import networkx as nx

    class NetworkXGraph(GraphWrapper, abstract=Graph):
        def __init__(
            self, nx_graph, node_weight_label="weight", edge_weight_label="weight"
        ):
            self.value = nx_graph
            self.node_weight_label = node_weight_label
            self.edge_weight_label = edge_weight_label
            self._assert_instance(nx_graph, nx.Graph)

        def _determine_dtype(self, all_values):
            all_types = {type(v) for v in all_values}
            if not all_types or (all_types - {float, int, bool}):
                return "str"
            for type_ in (float, int, bool):
                if type_ in all_types:
                    return str(type_.__name__)

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {"is_directed"} - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = obj.value.is_directed()

                # slow properties, only compute if asked
                slow_props = props - ret.keys()
                if {"node_type", "node_dtype"} & slow_props:
                    node_values = set()
                    for node in obj.value.nodes(data=True):
                        attrs = node[-1]
                        try:
                            node_values.add(attrs[obj.node_weight_label])
                        except KeyError:
                            node_values = None
                            break
                    if node_values:
                        ret["node_type"] = "map"
                        if "node_dtype" in slow_props:
                            ret["dtype"] = obj._determine_dtype(node_values)
                    else:
                        ret["node_type"] = "set"
                        ret["node_dtype"] = None

                if {
                    "edge_type",
                    "edge_dtype",
                    "edge_has_negative_weights",
                } & slow_props:
                    edge_values = set()
                    for edge in obj.value.edges(data=True):
                        attrs = edge[-1]
                        try:
                            edge_values.add(attrs[obj.edge_weight_label])
                        except KeyError:
                            edge_values = None
                            break
                    if edge_values:
                        ret["edge_type"] = "map"
                        if (
                            "edge_dtype" in slow_props
                            or "edge_has_negative_weights" in slow_props
                        ):
                            ret["edge_dtype"] = obj._determine_dtype(edge_values)
                        if "edge_has_negative_weights" in slow_props:
                            if ret["edge_dtype"] in {"bool", "str"}:
                                neg_weights = None
                            else:
                                min_val = min(edge_values)
                                if min_val < 0:
                                    neg_weights = True
                                else:
                                    neg_weights = False

                            ret["edge_has_negative_weights"] = neg_weights
                    else:
                        ret["edge_type"] = "set"
                        ret["edge_dtype"] = None
                        ret["edge_has_negative_weights"] = None

                return ret

            @classmethod
            def assert_equal(
                cls,
                obj1,
                obj2,
                aprops1,
                aprops2,
                cprops1,
                cprops2,
                *,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                g1 = obj1.value
                g2 = obj2.value
                # Compare
                assert g1.nodes() == g2.nodes(), f"{g1.nodes()} != {g2.nodes()}"
                assert g1.edges() == g2.edges(), f"{g1.edges()} != {g2.edges()}"

                if aprops1.get("node_type") == "map":
                    for n, d1 in g1.nodes(data=True):
                        d2 = g2.nodes[n]
                        val1 = d1[obj1.node_weight_label]
                        val2 = d2[obj2.node_weight_label]
                        if type(val1) is float:
                            assert math.isclose(
                                val1, val2, rel_tol=rel_tol, abs_tol=abs_tol
                            ), f"[{n}] {val1} not close to {val2}"
                        else:
                            assert val1 == val2, f"[{n}] {val1} != {val2}"

                if aprops1.get("edge_type") == "map":
                    for e1, e2, d1 in g1.edges(data=True):
                        d2 = g2.edges[(e1, e2)]
                        val1 = d1[obj1.edge_weight_label]
                        val2 = d2[obj2.edge_weight_label]

                    if type(val1) is float:
                        assert math.isclose(
                            val1, val2, rel_tol=rel_tol, abs_tol=abs_tol
                        ), f"{(e1, e2)} {val1} not close to {val2}"
                    else:
                        assert val1 == val2, f"{(e1, e2)} {val1} != {val2}"
