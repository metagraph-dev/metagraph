from typing import Set, Dict, Any
import math
from metagraph.types import NodeSet, NodeMap
from metagraph.wrappers import NodeSetWrapper, NodeMapWrapper


dtype_casting = {"str": str, "float": float, "int": int, "bool": bool}


class PythonNodeSet(NodeSetWrapper, abstract=NodeSet):
    def __init__(self, data):
        """
        data: set of node ids
        """
        super().__init__()
        self._assert_instance(data, set)
        self.value = data

    @property
    def num_nodes(self):
        return len(self.value)

    def copy(self):
        return PythonNodeSet(self.value.copy())

    def __contains__(self, key):
        return key in self.value

    class TypeMixin:
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
            rel_tol=None,
            abs_tol=None,
        ):
            v1, v2 = obj1.value, obj2.value
            assert len(v1) == len(v2), f"size mismatch: {len(v1)} != {len(v2)}"
            assert v1 == v2, f"node sets do not match"
            assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"


class PythonNodeMap(NodeMapWrapper, abstract=NodeMap):
    def __init__(self, data):
        """
        data: dict of node id: weight
        """
        super().__init__()
        self._assert_instance(data, dict)
        self.value = data

    def __getitem__(self, node_id):
        return self.value[node_id]

    @property
    def num_nodes(self):
        return len(self.value)

    def copy(self):
        return PythonNodeMap(self.value.copy())

    def __contains__(self, key):
        return key in self.value

    def _determine_dtype(self):
        types = set(type(val) for val in self.value.values())
        if not types or (types - {float, int, bool}):
            return "str"
        for type_ in (float, int, bool):
            if type_ in types:
                return str(type_.__name__)

    class TypeMixin:
        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: Set[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # slow properties, only compute if asked
            for prop in props - ret.keys():
                if prop == "dtype":
                    ret[prop] = obj._determine_dtype()

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
            d1, d2 = obj1.value, obj2.value
            if aprops1.get("dtype") == "float":
                assert (
                    not d1.keys() ^ d2.keys()
                ), f"Mismatched keys: {d1.keys() ^ d2.keys()}"
                assert all(
                    math.isclose(d1[key], d2[key], rel_tol=rel_tol, abs_tol=abs_tol)
                    for key in d1
                )
            else:
                assert d1 == d2
