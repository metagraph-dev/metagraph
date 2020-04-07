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


class PythonNodeMapping(Wrapper, abstract=NodeMapping):
    def __init__(self, data, src_labeled_index=None, dst_labeled_index=None):
        self._assert_instance(data, dict)
        self.data = data
        self.src_index = src_labeled_index
        self.dst_index = dst_labeled_index
