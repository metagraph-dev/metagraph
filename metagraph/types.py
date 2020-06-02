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
    conversions = {
        NodeSet: "to_nodeset",
    }

    @Wrapper.required_method
    def __getitem__(self, key):
        """Returns a scalar"""
        raise NotImplementedError()

    @Wrapper.required_property
    def num_nodes(self):
        raise NotImplementedError()

    @Wrapper.required_method
    def to_nodeset(self):
        raise NotImplementedError()


class NodeTable(AbstractType):
    conversions = {
        NodeSet: "to_nodeset",
    }

    @Wrapper.required_method
    def to_nodeset(self):
        raise NotImplementedError()


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
    conversions = {
        EdgeSet: "to_edgeset",
    }

    @Wrapper.required_method
    def to_edgeset(self):
        raise NotImplementedError()


class EdgeTable(AbstractType):
    properties = {"is_directed": [True, False]}
    conversions = {
        EdgeSet: "to_edgeset",
    }

    @Wrapper.required_method
    def to_edgeset(self):
        raise NotImplementedError()


del AbstractType, Wrapper
