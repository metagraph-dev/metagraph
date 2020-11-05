from typing import Set, Dict, Any
from metagraph.types import Graph, BipartiteGraph
from metagraph.wrappers import GraphWrapper, BipartiteGraphWrapper
from metagraph.plugins import has_networkx
import math


def _determine_dtype(all_values):
    all_types = {type(v) for v in all_values}
    if not all_types or (all_types - {float, int, bool}):
        # return "str"
        raise TypeError(f"unable to determine dtype, all_types={all_types}")
    for type_ in (float, int, bool):
        if type_ in all_types:
            return str(type_.__name__)


if has_networkx:
    import networkx as nx
    import copy

    class NetworkXGraph(GraphWrapper, abstract=Graph):
        def __init__(
            self,
            nx_graph,
            node_weight_label="weight",
            edge_weight_label="weight",
            *,
            aprops=None,
        ):
            super().__init__(aprops=aprops)
            self.value = nx_graph
            self.node_weight_label = node_weight_label
            self.edge_weight_label = edge_weight_label
            self._assert_instance(nx_graph, nx.Graph)

        def copy(self):
            return NetworkXGraph(
                copy.deepcopy(self.value),
                self.node_weight_label,
                self.edge_weight_label,
            )

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
                            ret["node_dtype"] = _determine_dtype(node_values)
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
                            ret["edge_dtype"] = _determine_dtype(edge_values)
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
                assert (
                    g1.nodes() == g2.nodes()
                ), f"node mismatch: {g1.nodes()} != {g2.nodes()}"
                assert (
                    g1.edges() == g2.edges()
                ), f"edge mismatch: {g1.edges()} != {g2.edges()}"

                if aprops1.get("node_type") == "map":
                    for n, d1 in g1.nodes(data=True):
                        d2 = g2.nodes[n]
                        val1 = d1[obj1.node_weight_label]
                        val2 = d2[obj2.node_weight_label]
                        if aprops1["node_dtype"] == "float":
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

                        if aprops1["edge_dtype"] == "float":
                            assert math.isclose(
                                val1, val2, rel_tol=rel_tol, abs_tol=abs_tol
                            ), f"{(e1, e2)} {val1} not close to {val2}"
                        else:
                            assert val1 == val2, f"{(e1, e2)} {val1} != {val2}"

    class NetworkXBipartiteGraph(BipartiteGraphWrapper, abstract=BipartiteGraph):
        def __init__(
            self,
            nx_graph,
            nodes,
            node_weight_label="weight",
            edge_weight_label="weight",
            *,
            aprops=None,
        ):
            """
            :param nx_graph:
            :param nodes: Tuple of sets nodes0 and nodes1
            :param node_weight_label:
            :param edge_weight_label:
            """
            super().__init__(aprops=aprops)
            self.value = nx_graph
            self.node_weight_label = node_weight_label
            self.edge_weight_label = edge_weight_label
            self._assert_instance(nx_graph, nx.Graph)
            if not hasattr(nodes, "__len__") or len(nodes) != 2:
                raise TypeError("nodes must have length of 2")
            self.nodes = (set(nodes[0]), set(nodes[1]))
            self._assert_instance(self.nodes[0], (set, list, tuple))
            self._assert_instance(self.nodes[1], (set, list, tuple))
            common_nodes = self.nodes[0] & self.nodes[1]
            if common_nodes:
                raise ValueError(
                    f"Node IDs found in both parts of the graph: {common_nodes}"
                )
            all_nodes = self.nodes[0] | self.nodes[1]
            unclaimed_nodes = nx_graph.nodes() - all_nodes
            if unclaimed_nodes:
                raise ValueError(
                    f"Node IDs found in graph, but not listed in either part: {unclaimed_nodes}"
                )

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
                if {"node0_type", "node0_dtype"} & slow_props:
                    node0_values = set()
                    for node_id in obj.nodes[0]:
                        attrs = obj.value.nodes[node_id]
                        try:
                            node0_values.add(attrs[obj.node_weight_label])
                        except KeyError:
                            node0_values = None
                            break
                    if node0_values:
                        ret["node0_type"] = "map"
                        if "node0_dtype" in slow_props:
                            ret["node0_dtype"] = _determine_dtype(node0_values)
                    else:
                        ret["node0_type"] = "set"
                        ret["node0_dtype"] = None

                if {"node1_type", "node1_dtype"} & slow_props:
                    node1_values = set()
                    for node_id in obj.nodes[1]:
                        attrs = obj.value.nodes[node_id]
                        try:
                            node1_values.add(attrs[obj.node_weight_label])
                        except KeyError:
                            node1_values = None
                            break
                    if node1_values:
                        ret["node1_type"] = "map"
                        if "node1_dtype" in slow_props:
                            ret["node1_dtype"] = _determine_dtype(node1_values)
                    else:
                        ret["node1_type"] = "set"
                        ret["node1_dtype"] = None

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
                            ret["edge_dtype"] = _determine_dtype(edge_values)
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
                assert (
                    obj1.nodes[0] == obj2.nodes[0]
                ), f"{obj1.nodes[0]} != {obj2.nodes[0]}"
                assert (
                    obj1.nodes[1] == obj2.nodes[1]
                ), f"{obj1.nodes[1]} != {obj2.nodes[1]}"
                assert g1.edges() == g2.edges(), f"{g1.edges()} != {g2.edges()}"

                if aprops1.get("node0_type") == "map":
                    for n in obj1.nodes[0]:
                        d1 = g1.nodes[n]
                        d2 = g2.nodes[n]
                        val1 = d1[obj1.node_weight_label]
                        val2 = d2[obj2.node_weight_label]
                        if aprops1["node0_dtype"] == "float":
                            assert math.isclose(
                                val1, val2, rel_tol=rel_tol, abs_tol=abs_tol
                            ), f"[{n}] {val1} not close to {val2}"
                        else:
                            assert val1 == val2, f"[{n}] {val1} != {val2}"

                if aprops1.get("node1_type") == "map":
                    for n in obj1.nodes[0]:
                        d1 = g1.nodes[n]
                        d2 = g2.nodes[n]
                        val1 = d1[obj1.node_weight_label]
                        val2 = d2[obj2.node_weight_label]
                        if aprops1["node1_dtype"] == "float":
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

                        if aprops1["edge_dtype"] == "float":
                            assert math.isclose(
                                val1, val2, rel_tol=rel_tol, abs_tol=abs_tol
                            ), f"{(e1, e2)} {val1} not close to {val2}"
                        else:
                            assert val1 == val2, f"{(e1, e2)} {val1} != {val2}"
