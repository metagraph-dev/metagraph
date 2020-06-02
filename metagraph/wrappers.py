from . import Wrapper
from .types import NodeSet, NodeMap, NodeTable, EdgeSet, EdgeMap, EdgeTable


class NodeSetWrapper(Wrapper, abstract=NodeSet, register=False):
    pass


class NodeMapWrapper(Wrapper, abstract=NodeMap, register=False):
    pass


class NodeTableWrapper(Wrapper, abstract=NodeTable, register=False):
    pass


class EdgeSetWrapper(Wrapper, abstract=EdgeSet, register=False):
    pass


class EdgeMapWrapper(Wrapper, abstract=EdgeMap, register=False):
    pass


class EdgeTableWrapper(Wrapper, abstract=EdgeTable, register=False):
    pass
