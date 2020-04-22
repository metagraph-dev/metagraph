import numpy as np
from metagraph import Wrapper, dtypes, SequentialNodes, IndexedNodes
from metagraph.types import Vector, Nodes, NodeMapping, Matrix, WEIGHT_CHOICES


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
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, cls.value_type):
            ret_val = cls()
            is_dense = obj.missing_mask is None
            dtype = dtypes.dtypes_simplified[obj.value.dtype]
            ret_val.abstract_instance = cls.abstract(is_dense=is_dense, dtype=dtype)
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")

    @classmethod
    def compare_objects(cls, obj1, obj2):
        if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
            raise TypeError("objects must be NumpyVector")

        if obj1.value.dtype != obj2.value.dtype:
            return False
        if obj1.value.shape != obj2.value.shape:
            return False
        # Remove missing values
        d1 = obj1.value if obj1.missing_mask is None else obj1.value[~obj1.missing_mask]
        d2 = obj2.value if obj2.missing_mask is None else obj2.value[~obj2.missing_mask]
        if d1.shape != d2.shape:
            return False
        # Check for alignment of missing masks
        if obj1.missing_mask is not None:
            if not (obj1.missing_mask == obj2.missing_mask).all():
                return False
        # Compare
        if issubclass(d1.dtype.type, np.floating):
            return np.isclose(d1, d2).all()
        else:
            return (d1 == d2).all()


class NumpyNodes(Wrapper, Nodes.Mixins, abstract=Nodes):
    """
    NumpyNodes stores data in verbose format with an entry in the array for every node
    If nodes are empty, a boolean missing_mask is provided
    """

    def __init__(self, data, *, weights=None, missing_mask=None, node_index=None):
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self.value = data
        self.missing_mask = missing_mask
        if missing_mask is not None and missing_mask.shape != data.shape:
            raise ValueError("missing_mask must be the same shape as data")
        self._dtype = dtypes.dtypes_simplified[data.dtype]
        self._weights = self._determine_weights(weights)
        self._node_index = node_index
        if node_index is not None and len(node_index) != len(data):
            raise ValueError(
                f"node_index size {len(node_index)} does not match data size {len(data)}"
            )

    def __getitem__(self, label):
        if self._node_index is None:
            return self.value[label]
        return self.value[self._node_index.bylabel(label)]

    def _determine_weights(self, weights=None):
        if weights is not None:
            if weights not in WEIGHT_CHOICES:
                raise ValueError(f"Illegal weights: {weights}")
            return weights

        if self._dtype == "str":
            return "any"
        values = (
            self.value if self.missing_mask is None else self.value[~self.missing_mask]
        )
        if self._dtype == "bool":
            if values.all():
                return "unweighted"
            return "non-negative"
        else:
            min_val = values.min()
            if min_val < 0:
                return "any"
            elif min_val == 0:
                return "non-negative"
            else:
                if self._dtype == "int" and min_val == 1 and values.max() == 1:
                    return "unweighted"
                return "positive"

    @property
    def num_nodes(self):
        return len(self.value)

    @property
    def node_index(self):
        if self._node_index is None:
            self._node_index = SequentialNodes(self.num_nodes)
        return self._node_index

    def rebuild_for_node_index(self, node_index):
        """
        Returns a new instance based on `node_index`
        """
        if self.num_nodes != len(node_index):
            raise ValueError(
                f"Size of node_index ({len(node_index)}) must match num_nodes ({self.num_nodes})"
            )

        data = self.value
        missing_mask = self.missing_mask
        if node_index != self.node_index:
            my_node_index = self.node_index
            my_node_index._verify_valid_conversion(node_index)
            index_converter = np.array(
                [my_node_index.bylabel(label) for label in node_index]
            )
            data = data[index_converter]
            if missing_mask is not None:
                missing_mask = missing_mask[index_converter]
        return NumpyNodes(
            data,
            weights=self._weights,
            missing_mask=missing_mask,
            node_index=node_index,
        )

    @classmethod
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, cls.value_type):
            ret_val = cls()
            ret_val.abstract_instance = cls.abstract(
                dtype=obj._dtype, weights=obj._weights
            )
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")

    @classmethod
    def compare_objects(cls, obj1, obj2):
        if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
            raise TypeError("objects must be NumpyNodes")

        if obj1.num_nodes != obj2.num_nodes:
            return False
        if obj1._dtype != obj2._dtype or obj1._weights != obj2._weights:
            return False
        # Convert to a common node indexing scheme
        try:
            obj2 = obj2.rebuild_for_node_index(obj1.node_index)
        except ValueError:
            return False
        # Remove missing values
        d1 = obj1.value if obj1.missing_mask is None else obj1.value[~obj1.missing_mask]
        d2 = obj2.value if obj2.missing_mask is None else obj2.value[~obj2.missing_mask]
        if len(d1) != len(d2):
            return False
        # Compare
        if obj1._dtype == "float":
            return np.isclose(d1, d2).all()
        else:
            return (d1 == d2).all()


class CompactNumpyNodes(Wrapper, Nodes.Mixins, abstract=Nodes):
    """
    CompactNumpyNodes only stores data for non-empty nodes
    num_nodes is determined by node_index, which must have an entry for all possible nodes, not just non-empty ones
    """

    def __init__(self, data, lookup, *, weights=None, node_index=None):
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self._assert_instance(lookup, dict)
        self._assert(len(data) == len(lookup), "size of data and lookup must match")
        self.value = data
        self.lookup = lookup
        self._dtype = dtypes.dtypes_simplified[data.dtype]
        self._weights = self._determine_weights(weights)
        self._node_index = node_index

    def __getitem__(self, label):
        idx = self.lookup[label]
        return self.value[idx]

    def _determine_weights(self, weights=None):
        if weights is not None:
            if weights not in WEIGHT_CHOICES:
                raise ValueError(f"Illegal weights: {weights}")
            return weights

        if self._dtype == "str":
            return "any"
        values = self.value
        if self._dtype == "bool":
            if values.all():
                return "unweighted"
            return "non-negative"
        else:
            min_val = values.min()
            if min_val < 0:
                return "any"
            elif min_val == 0:
                return "non-negative"
            else:
                if self._dtype == "int" and min_val == 1 and values.max() == 1:
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
            self._node_index = IndexedNodes.from_dict(self.lookup)
        return self._node_index

    def rebuild_for_lookup(self, lookup):
        if len(self.lookup) != len(lookup):
            raise ValueError(
                f"Size of lookup ({len(lookup)}) must match existing lookup ({len(self.lookup)})"
            )

        data = self.value

        if lookup != self.lookup:
            mismatch_keys = self.lookup.keys() ^ lookup.keys()
            if mismatch_keys:
                raise ValueError(
                    f"Unable to rebuild with mismatching keys: {mismatch_keys}"
                )

            index_converter = np.arange(len(lookup))
            for label, idx in lookup.items():
                index_converter[idx] = self.lookup[label]
            data = data[index_converter]

        return CompactNumpyNodes(
            data, lookup, weights=self._weights, node_index=self._node_index
        )

    @classmethod
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, cls.value_type):
            ret_val = cls()
            ret_val.abstract_instance = cls.abstract(
                dtype=obj._dtype, weights=obj._weights
            )
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")

    @classmethod
    def compare_objects(cls, obj1, obj2):
        if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
            raise TypeError("objects must be CompactNumpyNodes")

        if obj1.num_nodes != obj2.num_nodes:
            return False
        if obj1._dtype != obj2._dtype or obj1._weights != obj2._weights:
            return False
        if len(obj1.value) != len(obj2.value):
            return False
        # Convert to a common node ordering
        try:
            obj2 = obj2.rebuild_for_lookup(obj1.lookup)
        except ValueError:
            return False
        # Compare
        if obj1._dtype == "float":
            return np.isclose(obj1.value, obj2.value).all()
        else:
            return (obj1.value == obj2.value).all()


class NumpyNodeMapping(Wrapper, abstract=NodeMapping):
    def __init__(
        self,
        data,
        src_node_labels=None,
        dst_node_labels=None,
        # missing_value=_NONE_SPECIFIED,
    ):
        self.value = data
        # self.missing_value = missing_value
        self.src_node_labels = src_node_labels
        self.dst_node_labels = dst_node_labels


class NumpyMatrix(Wrapper, abstract=Matrix):
    def __init__(self, data, missing_mask=None, is_symmetric=None):
        if type(data) is np.matrix:
            data = np.array(data, copy=False)
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 2:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self.value = data
        nrows, ncols = data.shape
        self._is_square = nrows == ncols
        if is_symmetric is None:
            is_symmetric = nrows == ncols and (data.T == data).all().all()
        self._is_symmetric = is_symmetric
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
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, cls.value_type):
            ret_val = cls()
            is_dense = obj.missing_mask is None
            dtype = dtypes.dtypes_simplified[obj.value.dtype]
            ret_val.abstract_instance = cls.abstract(
                is_dense=is_dense,
                is_square=obj._is_square,
                is_symmetric=obj._is_symmetric,
                dtype=dtype,
            )
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")

    @classmethod
    def compare_objects(cls, obj1, obj2):
        if type(obj1) is not cls.value_type or type(obj2) is not cls.value_type:
            raise TypeError("objects must be NumpyMatrix")

        if obj1.value.dtype != obj2.value.dtype:
            return False
        if obj1.value.shape != obj2.value.shape:
            return False
        # Remove missing values
        d1 = obj1.value if obj1.missing_mask is None else obj1.value[~obj1.missing_mask]
        d2 = obj2.value if obj2.missing_mask is None else obj2.value[~obj2.missing_mask]
        if d1.shape != d2.shape:
            return False
        # Check for alignment of missing masks
        if obj1.missing_mask is not None:
            if not (obj1.missing_mask == obj2.missing_mask).all().all():
                return False
            # Compare 1-D
            if issubclass(d1.dtype.type, np.floating):
                return np.isclose(d1, d2).all()
            else:
                return (d1 == d2).all()
        else:
            # Compare 2-D
            if issubclass(d1.dtype.type, np.floating):
                return np.isclose(d1, d2).all().all()
            else:
                return (d1 == d2).all().all()
