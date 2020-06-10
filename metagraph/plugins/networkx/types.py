from typing import List, Dict, Any
from metagraph.types import EdgeSet, EdgeMap
from metagraph.wrappers import EdgeSetWrapper, EdgeMapWrapper
from metagraph.plugins import has_networkx
import math


if has_networkx:
    import networkx as nx

    class NetworkXEdgeSet(EdgeSetWrapper, abstract=EdgeSet):
        def __init__(self, nx_graph):
            self.value = nx_graph

        @classmethod
        def assert_equal(
            cls, obj1, obj2, props1, props2, *, rel_tol=None, abs_tol=None
        ):
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            g1 = obj1.value
            g2 = obj2.value
            # Compare
            assert (
                g1.nodes() == g2.nodes()
            ), f"node mismatch: {g1.nodes()} != {g2.nodes()}"
            assert (
                g1.edges() == g2.edges()
            ), f"edge mismatch: {g1.edges()} != {g2.edges()}"

    class NetworkXEdgeMap(EdgeMapWrapper, abstract=EdgeMap):
        def __init__(
            self, nx_graph, weight_label="weight",
        ):
            self.value = nx_graph
            self.weight_label = weight_label
            self._assert_instance(nx_graph, nx.Graph)

        def _determine_dtype(self, all_values):
            all_types = {type(v) for v in all_values}
            if not all_types or (all_types - {float, int, bool}):
                return "str"
            for type_ in (float, int, bool):
                if type_ in all_types:
                    return str(type_.__name__)

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"is_directed"} - ret.keys():
                if prop == "is_directed":
                    ret[prop] = obj.value.is_directed()

            # slow properties, only compute if asked
            slow_props = props - ret.keys()
            if "dtype" in slow_props or "has_negative_weights" in slow_props:
                all_values = set()
                for edge in obj.value.edges(data=True):
                    e_attrs = edge[-1]
                    value = e_attrs[obj.weight_label]
                    all_values.add(value)
                if "dtype" in slow_props:
                    ret["dtype"] = obj._determine_dtype(all_values)
                if "has_negative_weights" in slow_props:
                    if ret["dtype"] in {"bool", "str"}:
                        neg_weights = None
                    else:
                        min_val = min(all_values)
                        if min_val < 0:
                            neg_weights = True
                        else:
                            neg_weights = False
                    ret["has_negative_weights"] = neg_weights

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            g1 = obj1.value
            g2 = obj2.value
            # Compare
            assert g1.nodes() == g2.nodes(), f"{g1.nodes()} != {g2.nodes()}"
            assert g1.edges() == g2.edges(), f"{g1.edges()} != {g2.edges()}"

            for e1, e2, d1 in g1.edges(data=True):
                d2 = g2.edges[(e1, e2)]
                val1 = d1[obj1.weight_label]
                val2 = d2[obj2.weight_label]
                if type(val1) is float:
                    assert math.isclose(
                        val1, val2, rel_tol=rel_tol, abs_tol=abs_tol
                    ), f"{(e1, e2)} {val1} not close to {val2}"
                else:
                    assert val1 == val2, f"{(e1, e2)} {val1} != {val2}"
