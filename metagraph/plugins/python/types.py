import math
from metagraph import Wrapper, IndexedNodes
from metagraph.types import Nodes, NodeMapping, WEIGHT_CHOICES, DTYPE_CHOICES


dtype_casting = {
    "str": str,
    "float": float,
    "int": int,
    "bool": bool,
}


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
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, cls.value_type):
            ret_val = cls()
            ret_val.abstract_instance = Nodes(dtype=obj._dtype, weights=obj._weights)
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")

    @classmethod
    def compare_objects(
        cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True
    ):
        if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
            raise TypeError("objects must be PythonNodes")

        if check_values:
            if obj1._dtype != obj2._dtype or obj1._weights != obj2._weights:
                return False
            d1, d2 = obj1.value, obj2.value
            if obj1._dtype == "float":
                if d1.keys() ^ d2.keys():
                    return False
                return all(
                    math.isclose(d1[key], d2[key], rel_tol=rel_tol, abs_tol=abs_tol)
                    for key in d1
                )
            else:
                return d1 == d2
        else:
            return len(obj1.value) == len(obj2.value)


class PythonNodeMapping(Wrapper, abstract=NodeMapping):
    def __init__(self, data, src_labeled_index=None, dst_labeled_index=None):
        self._assert_instance(data, dict)
        self.data = data
        self.src_index = src_labeled_index
        self.dst_index = dst_labeled_index
