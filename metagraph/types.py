from . import AbstractType, Wrapper


DTYPE_CHOICES = ["str", "float", "int", "bool"]
WEIGHT_CHOICES = ["any", "non-negative", "positive", "unweighted"]


class Vector(AbstractType):
    properties = {"is_dense": [False, True], "dtype": DTYPE_CHOICES}


class NodeMap(AbstractType):
    properties = {"dtype": DTYPE_CHOICES, "weights": WEIGHT_CHOICES}

    @Wrapper.required_method
    def __getitem__(self, label):
        raise NotImplementedError()

    @Wrapper.required_property
    def num_nodes(self):
        raise NotImplementedError()

    @Wrapper.required_property
    def node_index(self):
        raise NotImplementedError()


class Matrix(AbstractType):
    properties = {
        "is_dense": [False, True],
        "is_square": [False, True],
        "is_symmetric": [False, True],
        "dtype": DTYPE_CHOICES,
    }


class DataFrame(AbstractType):
    pass


class Graph(AbstractType):
    properties = {
        "is_directed": [True, False],
        "dtype": DTYPE_CHOICES,
        "weights": WEIGHT_CHOICES,
    }

    @Wrapper.required_property
    def num_nodes(self):
        raise NotImplementedError()

    @Wrapper.required_property
    def node_index(self):
        raise NotImplementedError()


del AbstractType, Wrapper
