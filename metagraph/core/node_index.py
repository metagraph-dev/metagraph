import numpy as np


class InvalidNodeIndexConversion(Exception):
    pass


class IndexedNodes:
    def __init__(self, labels):
        if isinstance(labels, np.ndarray):
            if len(labels.shape) != 1:
                raise TypeError("labels must be list-like")
        elif not isinstance(labels, (tuple, list)):
            labels = tuple(labels)
        self._byindex = labels
        self._bylabel = {label: idx for idx, label in enumerate(labels)}

    @classmethod
    def from_dict(cls, labels_to_index):
        labels = [None] * len(labels_to_index)
        for label, idx in labels_to_index.items():
            labels[idx] = label
        # Avoid recomputation of dict
        ret_val = IndexedNodes(["dummy"])
        ret_val._byindex = labels
        ret_val._bylabel = labels_to_index.copy()
        return ret_val

    def __len__(self):
        return len(self._byindex)

    def __iter__(self):
        return iter(self._byindex)

    def __eq__(self, other):
        if type(other) is not IndexedNodes:
            return NotImplemented
        if len(other) != len(self):
            return False
        if isinstance(self._byindex, np.ndarray) and isinstance(
            other._byindex, np.ndarray
        ):
            return (self._byindex == other._byindex).all()
        else:
            return self._byindex == other._byindex

    def bylabel(self, label):
        return self._bylabel[label]

    def byindex(self, index):
        return self._byindex[index]

    def labels(self):
        return self._bylabel.keys()

    def _verify_valid_conversion(self, other):
        # If other does not contain all my labels, conversion is not possible
        missing_labels = self.labels() - other.labels()
        if missing_labels:
            raise ValueError(f"Labels missing from other: {missing_labels}")


class SequentialNodes(IndexedNodes):
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(range(self._size))

    def __eq__(self, other):
        if type(other) is not SequentialNodes:
            return NotImplemented
        return self._size == other._size

    def bylabel(self, label):
        return label

    def byindex(self, index):
        return index

    def labels(self):
        return set(iter(self))

    def _verify_valid_conversion(self, other):
        if type(other) is SequentialNodes and other._size < self._size:
            raise ValueError(
                f"other has {other._size} labels while this has {self._size}"
            )
        super()._verify_valid_conversion(other)
