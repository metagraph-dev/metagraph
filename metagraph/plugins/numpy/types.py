from typing import Set, Dict, Any
import numpy as np
from metagraph import dtypes, Wrapper, ConcreteType
from metagraph.types import Vector, Matrix, NodeSet, NodeMap
from metagraph.wrappers import NodeSetWrapper, NodeMapWrapper


class NumpyVectorType(ConcreteType, abstract=Vector):
    @classmethod
    def is_typeclass_of(cls, obj):
        """Is obj described by this type class?"""
        return isinstance(obj, np.ndarray) and len(obj.shape) == 1

    @classmethod
    def _compute_abstract_properties(
        cls, obj, props: Set[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        ret = known_props.copy()

        # fast properties
        for prop in {"dtype"} - ret.keys():
            if prop == "dtype":
                ret[prop] = dtypes.dtypes_simplified[obj.dtype]

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
        assert obj1.shape == obj2.shape, f"shape mismatch {obj1.shape} != {obj2.shape}"
        assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
        # Compare
        if issubclass(obj1.dtype.type, np.floating):
            assert np.isclose(obj1, obj2, rtol=rel_tol, atol=abs_tol).all()
        else:
            assert (obj1 == obj2).all()


class NumpyNodeSet(NodeSetWrapper, abstract=NodeSet):
    def __init__(self, nodes, *, aprops=None):
        super().__init__(aprops=aprops)
        self._assert_instance(nodes, (np.ndarray, list, tuple, set))
        if not isinstance(nodes, np.ndarray):
            if isinstance(nodes, set):
                nodes = tuple(nodes)  # np.array doesn't accept sets
            nodes = np.array(nodes)
        if len(nodes.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(nodes.shape)}")
        if not issubclass(nodes.dtype.type, np.integer):
            raise TypeError(f"Invalid dtype for NodeSet: {nodes.dtype}")
        # Ensure sorted
        nodes.sort()
        # Ensure no duplicates
        unique = np.diff(nodes) > 0
        if not unique.all():
            tmp = np.empty((unique.sum() + 1,), dtype=nodes.dtype)
            tmp[0] = nodes[0]
            tmp[1:] = nodes[1:][unique]
            nodes = tmp
        self.value = nodes

    @classmethod
    def from_mask(cls, mask, *, aprops=None):
        """
        The mask must be a boolean numpy array.
        NodeIds are based on position within the mask.
        """
        cls._assert_instance(mask, np.ndarray)
        cls._assert(mask.dtype == bool, "Only boolean masks are allowed")
        node_ids = np.flatnonzero(mask)
        return NumpyNodeSet(node_ids, aprops=aprops)

    def __len__(self):
        return len(self.value)

    def copy(self):
        aprops = NumpyNodeSet.Type.compute_abstract_properties(self, {})
        return NumpyNodeSet(self.value.copy(), aprops=aprops)

    def __iter__(self):
        return iter(self.value)

    def __contains__(self, key):
        index = np.searchsorted(self.value, key)
        if hasattr(index, "__len__"):
            return (self.value[index] == key).all()
        else:
            return self.value[index] == key

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
            assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
            assert len(obj1) == len(obj2), f"size mismatch: {len(obj1)} != {len(obj2)}"
            assert (obj1.value == obj2.value).all(), f"node sets do not match"


class NumpyNodeMap(NodeMapWrapper, abstract=NodeMap):
    """
    NumpyNodeMap stores data using numpy arrays. A mask of present values or
    a compact representation can be used.
    """

    def __init__(self, data, nodes=None, *, aprops=None):
        """
        data: values for each node
        nodes: array of node_ids corresponding ot elements in data

        If there are no missing nodes, nodes are not required. It will be assumed that node ids
        are sequential and the same size as `data`.
        """
        super().__init__(aprops=aprops)
        self._assert_instance(data, (np.ndarray, list, tuple))
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        if nodes is None:
            nodes = np.arange(len(data))
        else:
            self._assert_instance(nodes, (np.ndarray, list, tuple))
            if not isinstance(nodes, np.ndarray):
                nodes = np.array(nodes)
                if nodes.shape != data.shape:
                    raise TypeError(
                        f"Nodes must be same shape and size as data: {nodes.shape} != {data.shape}"
                    )
                if not issubclass(nodes.dtype.type, np.integer):
                    raise TypeError(f"Invalid dtype for NodeSet: {nodes.dtype}")
            # Ensure sorted
            if not np.all(np.diff(nodes) > 0):
                sorter = np.argsort(nodes)
                nodes = nodes[sorter]
                data = data[sorter]
            # Ensure no duplicates
            unique = np.diff(nodes) > 0
            if not unique.all():
                raise TypeError(f"Duplicate node ids found: {set(nodes[1:][~unique])}")

        self.value = data
        self.nodes = nodes

    @classmethod
    def from_mask(cls, data, mask, *, aprops=None):
        """
        Values in data are kept where mask is True.
        The mask must be a boolean numpy array.
        NodeIds are based on position within the mask.
        """
        cls._assert_instance(mask, np.ndarray)
        cls._assert(mask.dtype == bool, "Only boolean masks are allowed")
        data = data[mask]
        nodes = np.flatnonzero(mask)
        return NumpyNodeMap(data, nodes, aprops=aprops)

    def __len__(self):
        return len(self.value)

    def copy(self):
        aprops = NumpyNodeMap.Type.compute_abstract_properties(self, {})
        return NumpyNodeMap(self.value.copy(), nodes=self.nodes.copy(), aprops=aprops)

    def __contains__(self, key):
        index = np.searchsorted(self.nodes, key)
        if hasattr(index, "__len__"):
            return (self.nodes[index] == key).all()
        else:
            return self.nodes[index] == key

    def __getitem__(self, key):
        index = np.searchsorted(self.nodes, key)
        if hasattr(index, "__len__"):
            if not (self.nodes[index] == key).all():
                raise KeyError(f"nodes {key} are not all in the NodeMap")
        else:
            if self.nodes[index] != key:
                raise KeyError(f"node {key} is not in the NodeMap")
        return self.value[index]

    class TypeMixin:
        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: Set[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"dtype"} - ret.keys():
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
            assert len(obj1) == len(
                obj2
            ), f"length mismatch: {len(obj1)} != {len(obj2)}"
            assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"

            nodes1, vals1 = obj1.nodes, obj1.value
            nodes2, vals2 = obj2.nodes, obj2.value

            # Compare
            assert (nodes1 == nodes2).all(), f"node id mismatch: {nodes1} != {nodes2}"
            if issubclass(vals1.dtype.type, np.floating):
                assert np.isclose(vals1, vals2, rtol=rel_tol, atol=abs_tol).all()
            else:
                assert (vals1 == vals2).all()


class NumpyMatrixType(ConcreteType, abstract=Matrix):
    @classmethod
    def is_typeclass_of(cls, obj):
        """Is obj described by this type class?"""
        return isinstance(obj, np.ndarray) and len(obj.shape) == 2

    @classmethod
    def _compute_abstract_properties(
        cls, obj, props: Set[str], known_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        ret = known_props.copy()

        # fast properties
        for prop in {"is_dense", "is_square", "dtype"} - ret.keys():
            if prop == "dtype":
                ret[prop] = dtypes.dtypes_simplified[obj.dtype]

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
        assert obj1.shape == obj2.shape, f"shape mismatch: {obj1.shape} != {obj2.shape}"
        assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
        assert obj1.shape == obj2.shape, f"{obj1.shape} != {obj2.shape}"
        # Compare
        if issubclass(obj1.dtype.type, np.floating):
            assert np.isclose(obj1, obj2, rtol=rel_tol, atol=abs_tol).all().all()
        else:
            assert (obj1 == obj2).all().all()
