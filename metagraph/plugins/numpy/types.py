from typing import List, Dict, Any
import numpy as np
from metagraph import dtypes, Wrapper
from metagraph.types import Vector, Matrix, NodeSet, NodeMap
from metagraph.wrappers import NodeSetWrapper, NodeMapWrapper


class NumpyVector(Wrapper, abstract=Vector):
    def __init__(self, data, missing_mask=None):
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self.value = data
        self.missing_mask = missing_mask
        if missing_mask is not None:
            if missing_mask.dtype != bool:
                raise ValueError("missing_mask must have boolean type")
            if missing_mask.shape != data.shape:
                raise ValueError("missing_mask must be the same shape as data")

    def __len__(self):
        return len(self.value)

    @classmethod
    def compute_abstract_properties(
        cls, obj, props: List[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        cls._validate_abstract_props(props)
        is_dense = obj.missing_mask is None
        dtype = dtypes.dtypes_simplified[obj.value.dtype]
        return dict(is_dense=is_dense, dtype=dtype)

    @classmethod
    def assert_equal(cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0):
        assert (
            obj1.value.shape == obj2.value.shape
        ), f"{obj1.value.shape} != {obj2.value.shape}"
        # Remove missing values
        d1 = obj1.value if obj1.missing_mask is None else obj1.value[~obj1.missing_mask]
        d2 = obj2.value if obj2.missing_mask is None else obj2.value[~obj2.missing_mask]
        assert d1.shape == d2.shape, f"{d1.shape} != {d2.shape}"
        # Check for alignment of missing masks
        if obj1.missing_mask is not None:
            mask_alignment = obj1.missing_mask == obj2.missing_mask
            assert mask_alignment.all(), f"{mask_alignment}"
        # Compare
        if issubclass(d1.dtype.type, np.floating):
            assert np.isclose(d1, d2, rtol=rel_tol, atol=abs_tol).all()
        else:
            assert (d1 == d2).all()


class NumpyNodeMap(NodeMapWrapper, abstract=NodeMap):
    """
    NumpyNodeMap stores data in verbose format with an entry in the array for every node
    If nodes are empty, a boolean missing_mask is provided
    """

    def __init__(self, data, *, missing_mask=None):
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self.value = data
        self.missing_mask = missing_mask
        if missing_mask is not None and missing_mask.shape != data.shape:
            raise ValueError("missing_mask must be the same shape as data")

    def __getitem__(self, node_id):
        if self.missing_mask:
            if self.missing_mask[node_id]:
                raise ValueError(f"node {node_id} is not in the NodeMap")
        return self.value[node_id]

    def to_nodeset(self):
        from ..python.types import PythonNodeSet

        values = self.value
        if self.missing_mask is not None:
            values = values[~self.missing_mask]
        return PythonNodeSet(set(values))

    def _determine_weights(self, dtype):
        if dtype == "str":
            return "any"
        values = (
            self.value if self.missing_mask is None else self.value[~self.missing_mask]
        )
        if dtype == "bool":
            return "non-negative"
        else:
            min_val = values.min()
            if min_val < 0:
                return "any"
            elif min_val == 0:
                return "non-negative"
            else:
                return "positive"

    @property
    def num_nodes(self):
        if self.missing_mask is not None:
            # Count number of False in the missing mask
            return (~self.missing_mask).sum()
        return len(self.value)

    @classmethod
    def compute_abstract_properties(
        cls, obj, props: List[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        cls._validate_abstract_props(props)

        # fast properties
        ret = {"dtype": dtypes.dtypes_simplified[obj.value.dtype]}

        # slow properties, only compute if asked
        if "weights" in props:
            ret["weights"] = obj._determine_weights(ret["dtype"])

        return ret

    @classmethod
    def assert_equal(cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0):
        assert obj1.num_nodes == obj2.num_nodes, f"{obj1.num_nodes} != {obj2.num_nodes}"
        # Remove missing values
        d1 = obj1.value if obj1.missing_mask is None else obj1.value[~obj1.missing_mask]
        d2 = obj2.value if obj2.missing_mask is None else obj2.value[~obj2.missing_mask]
        assert len(d1) == len(d2), f"{len(d1)} != {len(d2)}"
        # Compare
        if issubclass(d1.dtype.type, np.floating):
            assert np.isclose(d1, d2, rtol=rel_tol, atol=abs_tol).all()
        else:
            assert (d1 == d2).all()


class CompactNumpyNodeMap(NodeMapWrapper, abstract=NodeMap):
    """
    CompactNumpyNodeMap only stores data for non-empty nodes
    """

    # TODO: make this style more general with a separate mapper including array of node_ids plus dict of {node_id: pos}
    def __init__(self, data, node_lookup):
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self._assert_instance(node_lookup, dict)
        self._assert(
            len(data) == len(node_lookup), "size of data and node_lookup must match"
        )
        self.value = data
        self.lookup = node_lookup

    def __getitem__(self, node_id):
        pos = self.lookup[node_id]
        return self.value[pos]

    @property
    def num_nodes(self):
        return len(self.lookup)

    def to_nodeset(self):
        from ..python.types import PythonNodeSet

        return PythonNodeSet(set(self.lookup))

    @classmethod
    def compute_abstract_properties(
        cls, obj, props: List[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        cls._validate_abstract_props(props)

        # fast properties
        ret = {"dtype": dtypes.dtypes_simplified[obj.value.dtype]}

        # slow properties, only compute if asked
        if "weights" in props:
            if ret["dtype"] == "str":
                weights = "any"
            elif ret["dtype"] == "bool":
                weights = "non-negative"
            else:
                min_val = obj.value.min()
                if min_val < 0:
                    weights = "any"
                elif min_val == 0:
                    weights = "non-negative"
                else:
                    weights = "positive"
            ret["weights"] = weights

        return ret

    @classmethod
    def assert_equal(cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0):
        assert obj1.num_nodes == obj2.num_nodes, f"{obj1.num_nodes} != {obj2.num_nodes}"
        assert len(obj1.value) == len(
            obj2.value
        ), f"{len(obj1.value)} != {len(obj2.value)}"
        # Compare
        if issubclass(obj1.value.dtype.type, np.floating):
            assert np.isclose(obj1.value, obj2.value, rtol=rel_tol, atol=abs_tol).all()
        else:
            assert (obj1.value == obj2.value).all()


class NumpyMatrix(Wrapper, abstract=Matrix):
    def __init__(self, data, missing_mask=None):
        if type(data) is np.matrix:
            data = np.array(data, copy=False)
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 2:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self.value = data
        self.missing_mask = missing_mask
        if missing_mask is not None:
            if missing_mask.dtype != bool:
                raise ValueError("missing_mask must have boolean type")
            if missing_mask.shape != data.shape:
                raise ValueError("missing_mask must be the same shape as data")

    @property
    def shape(self):
        return self.value.shape

    @classmethod
    def compute_abstract_properties(
        cls, obj, props: List[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        cls._validate_abstract_props(props)

        # fast properties
        ret = {
            "is_dense": obj.missing_mask is None,
            "is_square": obj.value.shape[0] == obj.value.shape[1],
            "dtype": dtypes.dtypes_simplified[obj.value.dtype],
        }

        # slow properties, only compute if asked
        if "is_symmetric" in props:
            # TODO: make this dependent on the missing mask
            ret["is_symmetric"] = (
                ret["is_square"] and (obj.value.T == obj.value).all().all()
            )

        return ret

    @classmethod
    def assert_equal(cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0):
        assert (
            obj1.value.shape == obj2.value.shape
        ), f"{obj1.value.shape} != {obj2.value.shape}"
        # Remove missing values
        d1 = obj1.value if obj1.missing_mask is None else obj1.value[~obj1.missing_mask]
        d2 = obj2.value if obj2.missing_mask is None else obj2.value[~obj2.missing_mask]
        assert d1.shape == d2.shape, f"{d1.shape} != {d2.shape}"
        # Check for alignment of missing masks
        if obj1.missing_mask is not None:
            mask_alignment = obj1.missing_mask == obj2.missing_mask
            assert mask_alignment.all().all(), f"{mask_alignment}"
            # Compare 1-D
            if issubclass(d1.dtype.type, np.floating):
                assert np.isclose(d1, d2, rtol=rel_tol, atol=abs_tol).all()
            else:
                assert (d1 == d2).all()
        else:
            # Compare 2-D
            if issubclass(d1.dtype.type, np.floating):
                assert np.isclose(d1, d2, rtol=rel_tol, atol=abs_tol).all().all()
            else:
                assert (d1 == d2).all().all()
