from . import AbstractType, Wrapper


# Use in signatures when a node ID is required
class NodeID:
    def __repr__(self):
        return "NodeID"

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Do not attempt to create a NodeID. Simply pass in the node_id as an int"
        )


# Create a singleton object which masks the class
NodeID = NodeID()


DTYPE_CHOICES = ["str", "float", "int", "bool"]


class Vector(AbstractType):
    properties = {
        "is_dense": [False, True],
        "dtype": DTYPE_CHOICES,
    }


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
    properties = {
        "dtype": DTYPE_CHOICES,
    }
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
        "has_negative_weights": [True, False],
    }
    unambiguous_subcomponents = {EdgeSet}


class EdgeTable(AbstractType):
    properties = {"is_directed": [True, False]}
    unambiguous_subcomponents = {EdgeSet}


#################################
# Graphs
#################################
class Graph(AbstractType):
    properties = {
        "is_directed": [True, False],
        "node_type": ["set", "map", "table"],
        "node_dtype": DTYPE_CHOICES + [None],
        "edge_type": ["set", "map", "table"],
        "edge_dtype": DTYPE_CHOICES + [None],
        "edge_has_negative_weights": [True, False, None],
    }
    unambiguous_subcomponents = {NodeSet, EdgeSet}


class BipartiteGraph(AbstractType):
    properties = {
        "is_directed": [True, False],
        "node_type": [
            "set",
            "map",
            "table",
        ],  # should be a list so both node sets can have independent types
        "node_dtype": DTYPE_CHOICES + [None],
        "edge_type": ["set", "map", "table"],
        "edge_dtype": DTYPE_CHOICES + [None],
        "edge_has_negative_weights": [True, False, None],
    }
    unambiguous_subcomponents = {EdgeSet}


del AbstractType, Wrapper
