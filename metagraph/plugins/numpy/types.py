import numpy as np
from metagraph import Wrapper, dtypes, SequentialNodes
from metagraph.types import Vector, Nodes, NodeMapping, Matrix, WEIGHT_CHOICES


_NONE_SPECIFIED = object()  # sentinel value


class NumpyVector(Wrapper, abstract=Vector):
    def __init__(self, data, missing_value=_NONE_SPECIFIED):
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 1:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
        self.value = data
        self.missing_value = missing_value

    def __len__(self):
        return len(self.value)

    def get_missing_mask(self):
        """
        Returns an array of True/False where True indicates a missing value
        """
        if self.missing_value is _NONE_SPECIFIED:
            return np.zeros_like(self.value, dtype=np.bool)

        if self.missing_value != self.missing_value:
            # Special handling for np.nan which does not equal itself
            return np.isnan(self.value)
        else:
            return self.value == self.missing_value

    @classmethod
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, cls.value_type):
            ret_val = cls()
            is_dense = obj.missing_value is _NONE_SPECIFIED
            dtype = dtypes.dtypes_simplified[obj.value.dtype]
            ret_val.abstract_instance = cls.abstract(is_dense=is_dense, dtype=dtype)
            return ret_val
        else:
            raise TypeError(f"object not of type {cls.__name__}")


class NumpyNodes(Wrapper, abstract=Nodes):
    def __init__(
        self, data, *, weights=None, missing_value=_NONE_SPECIFIED, node_index=None
    ):
        self.value = data
        self._assert_instance(data, np.ndarray)
        self.missing_value = missing_value
        self._dtype = dtypes.dtypes_simplified[data.dtype]
        self._weights = self._determine_weights(weights)
        self._node_index = node_index

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
        values = self.value[~self.get_missing_mask()]
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

    def get_missing_mask(self):
        """
        Returns an array of True/False where True indicates a missing value
        """
        if self.missing_value is _NONE_SPECIFIED:
            return np.zeros_like(self.value, dtype=np.bool)

        if self.missing_value != self.missing_value:
            # Special handling for np.nan which does not equal itself
            return np.isnan(self.value)
        else:
            return self.value == self.missing_value

    @property
    def node_index(self):
        if self._node_index is None:
            self._node_index = SequentialNodes(len(self.value))
        return self._node_index

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


class NumpyNodeMapping(Wrapper, abstract=NodeMapping):
    def __init__(
        self,
        data,
        src_node_labels=None,
        dst_node_labels=None,
        missing_value=_NONE_SPECIFIED,
    ):
        self.value = data
        self.missing_value = missing_value
        self.src_node_labels = src_node_labels
        self.dst_node_labels = dst_node_labels

    def get_missing_mask(self):
        """
        Returns an array of True/False where True indicates a missing value
        """
        if self.missing_value is _NONE_SPECIFIED:
            return np.zeros_like(self.value, dtype=np.bool)

        if self.missing_value != self.missing_value:
            # Special handling for np.nan which does not equal itself
            return np.isnan(self.value)
        else:
            return self.value == self.missing_value


class NumpyMatrix(Wrapper, abstract=Matrix):
    def __init__(self, data, missing_value=_NONE_SPECIFIED, is_symmetric=None):
        if type(data) is np.matrix:
            data = np.array(data, copy=False)
        self.value = data
        self.missing_value = missing_value
        nrows, ncols = data.shape
        self._is_square = nrows == ncols
        if is_symmetric is None:
            is_symmetric = (data.T == data).all().all()
        self._is_symmetric = is_symmetric
        self._assert_instance(data, np.ndarray)
        if len(data.shape) != 2:
            raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")

    def get_missing_mask(self):
        """
        Returns an array of True/False where True indicates a missing value
        """
        if self.missing_value is _NONE_SPECIFIED:
            return np.zeros_like(self.value, dtype=np.bool)

        if self.missing_value != self.missing_value:
            # Special handling for np.nan which does not equal itself
            return np.isnan(self.value)
        else:
            return self.value == self.missing_value

    @property
    def shape(self):
        return self.value.shape

    @classmethod
    def get_type(cls, obj):
        """Get an instance of this type class that describes obj"""
        if isinstance(obj, cls.value_type):
            ret_val = cls()
            is_dense = obj.missing_value is _NONE_SPECIFIED
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
