from typing import List, Dict, Any
import numpy as np
from metagraph import dtypes, Wrapper
from metagraph.types import Vector, Matrix, NodeSet, NodeMap
from metagraph.wrappers import NodeSetWrapper, NodeMapWrapper


class NumpyVector(Wrapper, abstract=Vector):
    def __init__(self, data, mask=None):
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self.value = data
        self.mask = mask
        if mask is not None:
            if mask.dtype != bool:
                raise ValueError("mask must have boolean type")
            if mask.shape != data.shape:
                raise ValueError("mask must be the same shape as data")

    def __len__(self):
        return len(self.value)

    class TypeMixin:
        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"is_dense", "dtype"} - ret.keys():
                if prop == "is_dense":
                    ret[prop] = obj.mask is None
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[obj.value.dtype]

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
            assert (
                obj1.value.shape == obj2.value.shape
            ), f"{obj1.value.shape} != {obj2.value.shape}"
            assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
            # Remove missing values
            d1 = obj1.value if obj1.mask is None else obj1.value[obj1.mask]
            d2 = obj2.value if obj2.mask is None else obj2.value[obj2.mask]
            assert d1.shape == d2.shape, f"{d1.shape} != {d2.shape}"
            # Check for alignment of masks
            if obj1.mask is not None:
                mask_alignment = obj1.mask == obj2.mask
                assert mask_alignment.all(), f"{mask_alignment}"
            # Compare
            if issubclass(d1.dtype.type, np.floating):
                assert np.isclose(d1, d2, rtol=rel_tol, atol=abs_tol).all()
            else:
                assert (d1 == d2).all()


class NumpyNodeMap(NodeMapWrapper, abstract=NodeMap):
    """
    NumpyNodeMap stores data using numpy arrays. A mask of present values or
    a compact representation can be used.
    """

    def __init__(self, data, *, mask=None, node_ids=None):
        """
        data: values for each node
        mask: True for each node, False if node is missing
        node_ids: array of node_ids which are not empty
        
        If providing mask, data must be as long as mask. Values which correspond to False in the mask are ignored.
        If providing node_ids, data must be the same length as node_ids.
        
        Provide either mask or node_ids, not both.
        If there are not missing nodes, mask and node_ids are not required.
        """
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self.value = data
        self.mask = None
        self.id2pos = None
        self.pos2id = None

        # Check input data
        if mask is not None:
            if node_ids is not None:
                raise ValueError("Cannot provide both mask and node_ids")
            self._assert_instance(mask, np.ndarray)
            if mask.shape != data.shape:
                raise ValueError("mask must be the same shape as data")
            self.mask = mask
        elif node_ids is not None:
            if isinstance(node_ids, dict):
                id2pos = node_ids
                pos2id = np.empty((len(node_ids),), dtype=int)
                for node_id, pos in node_ids.items():
                    pos2id[pos] = node_id
            elif isinstance(node_ids, np.ndarray):
                pos2id = node_ids
                id2pos = {node_id: pos for pos, node_id in enumerate(node_ids)}
            else:
                raise ValueError(f"Invalid type for node_ids: {type(node_ids)}")
            self.id2pos = id2pos
            self.pos2id = pos2id
            # Ensure all node ids are monotonically increasing
            self._assert(np.all(np.diff(pos2id) > 0), "Node IDs must be ordered")
            self._assert(
                len(id2pos) == len(data), f"node_ids must be the same length as data"
            )

    def __getitem__(self, node_id):
        if self.mask is not None:
            if self.mask[node_id]:
                raise ValueError(f"node {node_id} is not in the NodeMap")
        elif self.id2pos is not None:
            if node_id not in self.id2pos:
                raise ValueError(f"node {node_id} is not in the NodeMap")
            pos = self.id2pos[node_id]
            return self.value[pos]
        return self.value[node_id]

    @property
    def num_nodes(self):
        if self.mask is not None:
            # Count number of True in the mask
            return self.mask.sum()
        # This covers the sequential and compact cases
        return len(self.value)

    class TypeMixin:
        allowed_props = {"is_compact": [True, False]}

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"dtype"} - ret.keys():
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[obj.value.dtype]

            return ret

        @classmethod
        def _compute_concrete_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            return {"is_compact": obj.id2pos is not None}

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
            assert (
                obj1.num_nodes == obj2.num_nodes
            ), f"{obj1.num_nodes} != {obj2.num_nodes}"
            assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"

            # Standardize obj1
            if obj1.mask is not None:
                vals1 = obj1.value[obj1.mask]
                nodes1 = np.flatnonzero(obj1.mask)
            elif obj1.id2pos is not None:
                vals1 = obj1.value
                nodes1 = obj1.pos2id
            else:
                vals1 = obj1.value
                nodes1 = np.array([])
            # Standardize obj2
            if obj2.mask is not None:
                vals2 = obj2.value[obj2.mask]
                nodes2 = np.flatnonzero(obj2.mask)
            elif obj2.id2pos is not None:
                vals2 = obj2.value
                nodes2 = obj2.pos2id
            else:
                vals2 = obj2.value
                nodes2 = np.array([])

            # Compare
            assert len(nodes1) == len(
                nodes2
            ), f"node ids not same length: {len(nodes1)} != {len(nodes2)}"
            assert (nodes1 == nodes2).all(), f"node id mismatch: {nodes1} != {nodes2}"
            assert len(vals1) == len(
                vals2
            ), f"non-empty value length mismatch: {len(vals1)} != {len(vals2)}"
            if issubclass(vals1.dtype.type, np.floating):
                assert np.isclose(vals1, vals2, rtol=rel_tol, atol=abs_tol).all()
            else:
                assert (vals1 == vals2).all()


class NumpyMatrix(Wrapper, abstract=Matrix):
    def __init__(self, data, mask=None):
        if type(data) is np.matrix:
            data = np.array(data, copy=False)
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 2:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self.value = data
        self.mask = mask
        if mask is not None:
            if mask.dtype != bool:
                raise ValueError("mask must have boolean type")
            if mask.shape != data.shape:
                raise ValueError("mask must be the same shape as data")

    @property
    def shape(self):
        return self.value.shape

    class TypeMixin:
        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"is_dense", "is_square", "dtype"} - ret.keys():
                if prop == "is_dense":
                    ret[prop] = obj.mask is None
                if prop == "is_square":
                    ret[prop] = obj.value.shape[0] == obj.value.shape[1]
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[obj.value.dtype]

            # slow properties, only compute if asked
            for prop in props - ret.keys():
                if prop == "is_symmetric":
                    # TODO: make this dependent on the mask
                    ret[prop] = (
                        ret["is_square"] and (obj.value.T == obj.value).all().all()
                    )

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
            assert (
                obj1.value.shape == obj2.value.shape
            ), f"{obj1.value.shape} != {obj2.value.shape}"
            assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
            # Remove missing values
            d1 = obj1.value if obj1.mask is None else obj1.value[obj1.mask]
            d2 = obj2.value if obj2.mask is None else obj2.value[obj2.mask]
            assert d1.shape == d2.shape, f"{d1.shape} != {d2.shape}"
            # Check for alignment of masks
            if obj1.mask is not None:
                mask_alignment = obj1.mask == obj2.mask
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
