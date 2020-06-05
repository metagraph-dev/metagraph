from typing import List, Dict, Any
import math
from metagraph.types import NodeSet, NodeMap
from metagraph.wrappers import NodeSetWrapper, NodeMapWrapper


dtype_casting = {"str": str, "float": float, "int": int, "bool": bool}


class PythonNodeSet(NodeSetWrapper, abstract=NodeSet):
    def __init__(self, data):
        """
        data: set of node ids
        """
        self._assert_instance(data, set)
        self.value = data

    @classmethod
    def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=None, abs_tol=None):
        v1, v2 = obj1.value, obj2.value
        assert len(v1) == len(v2), f"size mismatch: {len(v1)} != {len(v2)}"
        assert v1 == v2, f"node sets do not match"
        assert props1 == props2, f"property mismatch: {props1} != {props2}"


class PythonNodeMap(NodeMapWrapper, abstract=NodeMap):
    def __init__(self, data):
        """
        data: dict of node id: weight
        """
        self._assert_instance(data, dict)
        self.value = data

    def __getitem__(self, node_id):
        return self.value[node_id]

    @property
    def num_nodes(self):
        return len(self.value)

    def _determine_dtype(self):
        types = set(type(val) for val in self.value.values())
        if not types or (types - {float, int, bool}):
            return "str"
        for type_ in (float, int, bool):
            if type_ in types:
                return str(type_.__name__)

    def _determine_weights(self, dtype):
        if dtype == "str":
            return "any"
        elif dtype == "bool":
            return "non-negative"
        else:
            min_val = min(self.value.values())
            if min_val < 0:
                return "any"
            elif min_val == 0:
                return "non-negative"
            else:
                return "positive"

    @classmethod
    def _compute_abstract_properties(
        cls, obj, props: List[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        ret = known_props.copy()

        # slow properties, only compute if asked
        slow_props = props - ret.keys()
        if "weights" in slow_props and "dtype" not in ret:
            ret["dtype"] = obj._determine_dtype()
        for prop in props - ret.keys():
            if prop == "weights":
                ret["weights"] = obj._determine_weights(ret["dtype"])

        return ret

    @classmethod
    def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
        assert props1 == props2, f"property mismatch: {props1} != {props2}"
        d1, d2 = obj1.value, obj2.value
        if props1["dtype"] == "float":
            assert (
                not d1.keys() ^ d2.keys()
            ), f"Mismatched keys: {d1.keys() ^ d2.keys()}"
            assert all(
                math.isclose(d1[key], d2[key], rel_tol=rel_tol, abs_tol=abs_tol)
                for key in d1
            )
        else:
            assert d1 == d2
