from ... import PluginRegistry, ConcreteType, dtypes
from ..abstract_types import SparseVectorType

reg = PluginRegistry("metagraph_core")


_dtype_mapper = {
    bool: dtypes.BOOL,
    int: dtypes.INT64,
    float: dtypes.FLOAT64,
}


class PythonSparseVector:
    def __init__(self, data, size=None):
        """
        data: dict of node: weight
        size: total number of possible nodes
              if None, computes the size as the max index found
        """
        super().__init__()
        self.obj = data
        if size is None:
            size = max(data.keys())
        self.size = size
        self._dtype = None
        assert isinstance(data, dict)

    def __len__(self):
        return self.size

    @property
    def dtype(self):
        if self._dtype is None:
            types = set(type(val) for val in self.obj.values())
            if not types:
                return None
            for type_ in (float, int, bool):
                if type_ in types:
                    break
            else:
                raise Exception(f"Invalid type: {types}")
            self._dtype = _dtype_mapper[type_]
        return self._dtype


@reg.register
class PythonSparseVectorType(ConcreteType):
    name = "PythonSparseVector"
    abstract = SparseVectorType
    value_class = PythonSparseVector
