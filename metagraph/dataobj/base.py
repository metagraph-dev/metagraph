from collections.abc import MutableSequence


class BaseDataObj:
    @classmethod
    def property_schema(cls):
        # TODO: flesh out properties
        #       we will need:
        #         - enumerated properties (list of possible values)
        #         - typed properties (bool, int, float)
        #      allow nesting? (e.g. shape is a 2-tuple of type int)
        return {}

    def to(self, data_class, **props):
        from ..translator.base import (
            _translator_registry,
        )  # import here to avoid circular import

        my_class = type(self)
        key = (my_class, data_class)
        try:
            translator_func = _translator_registry[key]
            return translator_func(self, **props)
        except KeyError:
            valid_destinations = [
                oc for ic, oc in _translator_registry.keys() if ic == my_class
            ]
            if valid_destinations:
                print(
                    f"No translator registered for {my_class.__name__} -> {data_class.__name__}"
                )
                print(f"Translators exist for:")
                for dest_class in valid_destinations:
                    print(f"\t{my_class.__name__} -> {dest_class.__name__}")
            else:
                print(f"No translators have been registered for {my_class.__name__}")
            raise


class DataList(MutableSequence):
    def __init__(self, data_class):
        self._data_class = data_class
        self._data = []

    @classmethod
    def from_items(cls, items):
        item = items[0]
        dl = cls(item.__class__)
        dl.extend(items)
        return dl

    def __repr__(self):
        return f"List<{self._data_class.__name__}> of length {len(self)}"

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, val):
        self._check_type(val)
        self._data[key] = val

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def insert(self, idx, val):
        self._check_type(val)
        return self._data.insert(idx, val)

    def _check_type(self, val):
        if type(val) != self._data_class:
            raise TypeError(
                f"Wrong type for DataList: {type(val).__name__}; expected {self._data_class.__name__}"
            )


########################################
# Basic categories
#
# These are not strictly required,
# but help group concrete data objects
########################################
class Scalar(BaseDataObj):
    pass


class SparseArray(BaseDataObj):
    pass


class DenseArray(BaseDataObj):
    def __init__(self, missing_value):
        self.missing_value = missing_value


class SparseMatrix(BaseDataObj):
    pass


class DenseMatrix(BaseDataObj):
    def __init__(self, missing_value):
        self.missing_value = missing_value


class DataFrame(BaseDataObj):
    pass


class Graph(BaseDataObj):
    pass


class WeightedGraph(Graph):
    pass


class AdjacencyMatrix(Graph):
    def __init__(self, transposed=False):
        """
        in-degree is found by summing the column; out-degree is found by summing the row
        If the underlying matrix has these properties reversed, set transposed=True
        """
        self.transposed = transposed


class WeightedAdjacencyMatrix(AdjacencyMatrix):
    pass


class IncidenceMatrix(Graph):
    def __init__(self, transposed=False):
        """
        nodes are rows; edges are columns
        If the underlying matrix has these properties reversed, set transposed=True
        """
        self.transposed = transposed


class EdgeListDF(Graph):
    def __init__(self, source_label="source", dest_label="destination"):
        self.source_label = source_label
        self.dest_label = dest_label


class WeightedEdgeListDF(EdgeListDF):
    def __init__(
        self, source_label="source", dest_label="destination", weight_label="weights"
    ):
        super().__init__(source_label, dest_label)
        self.weight_label = weight_label
