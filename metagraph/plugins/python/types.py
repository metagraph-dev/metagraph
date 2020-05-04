from typing import List, Dict, Any
import math
from metagraph import Wrapper, IndexedNodes
from metagraph.types import Nodes, NodeMapping, WEIGHT_CHOICES, DTYPE_CHOICES


dtype_casting = {"str": str, "float": float, "int": int, "bool": bool}


class PythonNodes(Wrapper, abstract=Nodes):
    def __init__(self, data, *, dtype=None, weights=None, node_index=None):
        """
        data: dict of node: weight
        """
        self._assert_instance(data, dict)
        self.value = data
        self._dtype = self._determine_dtype(dtype)
        self._weights = self._determine_weights(weights)
        self._node_index = node_index

    def __getitem__(self, label):
        return self.value[label]

    def _determine_dtype(self, dtype):
        if dtype is not None:
            if dtype not in DTYPE_CHOICES:
                raise ValueError(f"Illegal dtype: {dtype}")
            return dtype

        types = set(type(val) for val in self.value.values())
        if not types or (types - {float, int, bool}):
            return "str"
        for type_ in (float, int, bool):
            if type_ in types:
                return str(type_.__name__)

    def _determine_weights(self, weights):
        if weights is not None:
            if weights not in WEIGHT_CHOICES:
                raise ValueError(f"Illegal weights: {weights}")
            return weights

        if self._dtype == "str":
            return "any"
        if self._dtype == "bool":
            if set(self.value.values()) == {True}:
                return "unweighted"
            return "non-negative"
        else:
            min_val = min(self.value.values())
            if min_val < 0:
                return "any"
            elif min_val == 0:
                return "non-negative"
            else:
                if self._dtype == "int" and set(self.value.values()) == {1}:
                    return "unweighted"
                return "positive"

    @property
    def num_nodes(self):
        if self._node_index is None:
            return len(self.value)
        return len(self._node_index)

    @property
    def node_index(self):
        if self._node_index is None:
            nodes = tuple(self.value.keys())
            if type(nodes[0]) == int:
                nodes = sorted(nodes)
            self._node_index = IndexedNodes(nodes)
        return self._node_index

    @classmethod
    def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
        cls._validate_abstract_props(props)
        return dict(dtype=obj._dtype, weights=obj._weights)

    @classmethod
    def assert_equal(cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True):
        assert (
            type(obj1) is cls.value_type
        ), f"obj1 must be PythonNodes, not {type(obj1)}"
        assert (
            type(obj2) is cls.value_type
        ), f"obj2 must be PythonNodes, not {type(obj2)}"

        if check_values:
            assert obj1._dtype == obj2._dtype, f"{obj1._dtype} != {obj2._dtype}"
            assert obj1._weights == obj2._weights, f"{obj1._weights} != {obj2._weights}"
            d1, d2 = obj1.value, obj2.value
            if obj1._dtype == "float":
                assert (
                    not d1.keys() ^ d2.keys()
                ), f"Mismatched keys: {d1.keys() ^ d2.keys()}"
                assert all(
                    math.isclose(d1[key], d2[key], rel_tol=rel_tol, abs_tol=abs_tol)
                    for key in d1
                )
            else:
                assert d1 == d2
        else:
            assert len(obj1.value) == len(
                obj2.value
            ), f"{len(obj1.value)} != {len(obj2.value)}"


class PythonNodeMapping(Wrapper, abstract=NodeMapping):
    def __init__(self, data, src_labeled_index=None, dst_labeled_index=None):
        self._assert_instance(data, dict)
        self.data = data
        self.src_index = src_labeled_index
        self.dst_index = dst_labeled_index
