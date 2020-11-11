from typing import Set, Dict, Any
import math
from metagraph.types import NodeSet, NodeMap
from metagraph import ConcreteType


dtype_casting = {"str": str, "float": float, "int": int, "bool": bool}


class PythonNodeSetType(ConcreteType, abstract=NodeSet):
    value_type = set

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
        v1, v2 = obj1, obj2
        assert len(v1) == len(v2), f"size mismatch: {len(v1)} != {len(v2)}"
        assert v1 == v2, f"node sets do not match: {v1 ^ v2}"
        assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"


class PythonNodeMapType(ConcreteType, abstract=NodeMap):
    value_type = dict

    saved = {}

    @classmethod
    def _compute_abstract_properties(
        cls, obj, props: Set[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        ret = known_props.copy()

        # slow properties, only compute if asked
        for prop in props - ret.keys():
            if prop == "dtype":
                # Determine dtype
                types = set(type(val) for val in obj.values())
                if not types or (types - {float, int, bool}):
                    raise TypeError(f"Unable to compute dtype for types {types}")
                for type_ in (float, int, bool):
                    if type_ in types:
                        ret[prop] = str(type_.__name__)
                        break

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
        d1, d2 = obj1, obj2
        assert not d1.keys() ^ d2.keys(), f"Mismatched keys: {d1.keys() ^ d2.keys()}"
        if aprops1.get("dtype") == "float":
            for key in d1.keys() | d2.keys():
                assert math.isclose(
                    d1[key], d2[key], rel_tol=rel_tol, abs_tol=abs_tol
                ), f"Mismatch for node {key}: {d1[key]} != {d2[key]}"
        else:
            if d1 != d2:
                for key in d1.keys() | d2.keys():
                    assert (
                        d1[key] == d2[key]
                    ), f"Mismatch for node {key}: {d1[key]} != {d2[key]}"
