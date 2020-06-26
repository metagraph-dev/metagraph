class NodeLabels:
    """
    Bidirectional mapping from node_id to label

    node_id must be an int
    label must be hashable

    Behaves like a mapping from label to node_id
    Contains a `.ids` attribute which behaves like a mapping from node_id back to label
    Passing a list of labels or node_ids will return a list, allowing for easier handling of
        edges, which are simply a list of two nodes. The type must be list, not tuple to avoid
        any potential conflict with the label being a tuple. Lists are not hashable and therefore
        cannot be used as a valid label, so handling lists specially allows for clear disambiguation
        of meaning by the caller.

    Usage
    -----
    >>> node_labels = NodeLabels([0, 10, 42], ['A', 'B', 'C'])
    >>> node_labels['B']
    10
    >>> node_labels.ids[42]
    'C'
    >>> 'D' in node_labels
    False
    >>> 42 in node_labels.ids
    True
    >>> node_labels[['A', 'C']]
    [0, 42]
    >>> node_labels.ids[[42, 10]]
    ['C', 'B']
    """

    def __init__(self, node_ids, labels):
        if len(node_ids) != len(labels):
            raise ValueError(f"lengths must match: {len(node_ids)} != {len(labels)}")

        id2label = {}
        label2id = {}
        for nodeid, label in zip(node_ids, labels):
            if type(nodeid) is not int:
                raise TypeError(f"node ids must be int, not {type(nodeid)}")
            id2label[nodeid] = label
            label2id[label] = nodeid

        if len(id2label) != len(node_ids):
            raise ValueError("duplicate node ids")
        if len(label2id) != len(labels):
            raise ValueError("duplicate labels")

        self._id2label = id2label
        self._label2id = label2id

        self.ids = NodeLabels._ReverseMapper(self)

    @classmethod
    def from_dict(cls, mapping):
        if not hasattr(mapping, "items"):
            raise TypeError(f"mapping must be dict-like, not {type(mapping)}")
        if len(mapping) <= 0:
            raise ValueError("mapping is empty")

        # Find whether keys are ids or labels
        sample_key = next(iter(mapping))
        if type(sample_key) is int:
            node_ids, labels = zip(*mapping.items())
        else:
            labels, node_ids = zip(*mapping.items())
        return NodeLabels(node_ids, labels)

    def __eq__(self, other):
        if type(other) is not NodeLabels:
            return NotImplemented
        return self._label2id == other._label2id

    def __len__(self):
        return len(self._label2id)

    def __getitem__(self, label):
        if type(label) is list:
            return [self._label2id[x] for x in label]
        return self._label2id[label]

    def __contains__(self, item):
        return item in self._label2id

    class _ReverseMapper:
        def __init__(self, outer):
            self._outer = outer

        def __getitem__(self, node_id):
            if type(node_id) is list:
                return [self._outer._id2label[x] for x in node_id]
            return self._outer._id2label[node_id]

        def __contains__(self, node_id):
            return node_id in self._outer._id2label
