from typing import List, Dict, Any
from metagraph import ConcreteType, Wrapper, IndexedNodes
from metagraph.types import Graph, DTYPE_CHOICES, WEIGHT_CHOICES
from metagraph.plugins import has_networkx
from functools import partial
import operator
import math


if has_networkx:
    import networkx as nx

    class NetworkXGraph(Wrapper, abstract=Graph):
        def __init__(
            self,
            nx_graph,
            weight_label=None,
            *,
            weights=None,
            dtype=None,
            node_index=None,
        ):
            self.value = nx_graph
            self.weight_label = weight_label
            self._node_index = node_index
            self._assert_instance(nx_graph, nx.Graph)
            if weight_label is None:
                self._dtype = "bool"
                self._weights = "unweighted"
            else:
                all_values = set()
                for edge in nx_graph.edges(data=True):
                    e_attrs = edge[-1]
                    value = e_attrs[weight_label]
                    all_values.add(value)
                self._dtype = self._determine_dtype(dtype, all_values)
                self._weights = self._determine_weights(weights, all_values)

        def _determine_dtype(self, dtype, all_values):
            if dtype is not None:
                if dtype not in DTYPE_CHOICES:
                    raise ValueError(f"Illegal dtype: {dtype}")
                return dtype

            all_types = {type(v) for v in all_values}
            if not all_types or (all_types - {float, int, bool}):
                return "str"
            for type_ in (float, int, bool):
                if type_ in all_types:
                    return str(type_.__name__)

        def _determine_weights(self, weights, all_values):
            if weights is not None:
                if weights not in WEIGHT_CHOICES:
                    raise ValueError(f"Illegal weights: {weights}")
                return weights

            if self._dtype == "str":
                return "any"
            if self._dtype == "bool":
                if all_values == {True}:
                    return "unweighted"
                return "non-negative"
            else:
                min_val = min(all_values)
                if min_val < 0:
                    return "any"
                elif min_val == 0:
                    return "non-negative"
                else:
                    if self._dtype == "int" and all_values == {1}:
                        return "unweighted"
                    return "positive"

        @property
        def num_nodes(self):
            return self.value.number_of_nodes()

        @property
        def node_index(self):
            if self._node_index is None:
                nodes = tuple(self.value.nodes())
                if type(nodes[0]) == int:
                    nodes = sorted(nodes)
                self._node_index = IndexedNodes(nodes)
            return self._node_index

        @classmethod
        def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
            cls._validate_abstract_props(props)
            return dict(
                is_directed=obj.value.is_directed(),
                dtype=obj._dtype,
                weights=obj._weights,
            )

        @classmethod
        def assert_equal(
            cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True
        ):
            assert (
                type(obj1) is cls.value_type
            ), f"obj1 must be NetworkXGraph, not {type(obj1)}"
            assert (
                type(obj2) is cls.value_type
            ), f"obj2 must be NetworkXGraph, not {type(obj2)}"

            if check_values:
                assert obj1._dtype == obj2._dtype, f"{obj1._dtype} != {obj2._dtype}"
                assert (
                    obj1._weights == obj2._weights
                ), f"{obj1._weights} != {obj2._weights}"
            g1 = obj1.value
            g2 = obj2.value
            assert (
                g1.is_directed() == g2.is_directed()
            ), f"{g1.is_directed()} != {g2.is_directed()}"
            # Compare
            assert g1.nodes() == g2.nodes(), f"{g1.nodes()} != {g2.nodes()}"
            assert g1.edges() == g2.edges(), f"{g1.edges()} != {g2.edges()}"
            if check_values and obj1._weights != "unweighted":
                if obj1._dtype == "float":
                    comp = partial(math.isclose, rel_tol=rel_tol, abs_tol=abs_tol)
                    compstr = "close to"
                else:
                    comp = operator.eq
                    compstr = "equal to"

                for e1, e2, d1 in g1.edges(data=True):
                    d2 = g2.edges[(e1, e2)]
                    val1 = d1[obj1.weight_label]
                    val2 = d2[obj2.weight_label]
                    assert comp(val1, val2), f"{(e1, e2)} {val1} not {compstr} {val2}"
