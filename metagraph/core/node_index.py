import numpy as np


class IndexedNodes:
    def __init__(self, labels):
        if isinstance(labels, np.ndarray):
            if len(labels.shape) != 1:
                raise TypeError("labels must be list-like")
        elif not isinstance(labels, (tuple, list)):
            labels = tuple(labels)
        self._byindex = labels
        self._bylabel = {label: idx for idx, label in enumerate(labels)}

    def __len__(self):
        return len(self._byindex)

    def __iter__(self):
        return iter(self._byindex)

    def bylabel(self, label):
        return self._bylabel[label]

    def byindex(self, index):
        label = self._byindex[index]
        return label


class SequentialNodes(IndexedNodes):
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(range(self._size))

    def bylabel(self, label):
        return label

    def byindex(self, index):
        return index
