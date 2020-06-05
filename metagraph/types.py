from . import AbstractType, Wrapper


DTYPE_CHOICES = ["str", "float", "int", "bool"]
WEIGHT_CHOICES = ["any", "non-negative", "positive"]


class Vector(AbstractType):
    properties = {"is_dense": [False, True], "dtype": DTYPE_CHOICES}


class Matrix(AbstractType):
    properties = {
        "is_dense": [False, True],
        "is_square": [False, True],
        "dtype": DTYPE_CHOICES,
    }


class DataFrame(AbstractType):
    pass


#################################
# Nodes
#################################
class NodeSet(AbstractType):
    pass


class NodeMap(AbstractType):
    properties = {"dtype": DTYPE_CHOICES, "weights": WEIGHT_CHOICES}
    unambiguous_subcomponents = {NodeSet}

    @Wrapper.required_method
    def __getitem__(self, key):
        """Returns a scalar"""
        raise NotImplementedError()

    @Wrapper.required_property
    def num_nodes(self):
        raise NotImplementedError()


class NodeTable(AbstractType):
    unambiguous_subcomponents = {NodeSet}


#################################
# Edges
#################################
class EdgeSet(AbstractType):
    properties = {"is_directed": [True, False]}


class EdgeMap(AbstractType):
    properties = {
        "is_directed": [True, False],
        "dtype": DTYPE_CHOICES,
        "weights": WEIGHT_CHOICES,
    }
    unambiguous_subcomponents = {EdgeSet}


class EdgeTable(AbstractType):
    properties = {"is_directed": [True, False]}
    unambiguous_subcomponents = {EdgeSet}


del AbstractType, Wrapper
