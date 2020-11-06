from . import Wrapper
from .types import (
    NodeSet,
    NodeMap,
    # NodeTable,
    EdgeSet,
    EdgeMap,
    # EdgeTable,
    Graph,
    BipartiteGraph,
    GraphSageNodeEmbedding,
)
from typing import Set, Dict, Any


class NodeSetWrapper(Wrapper, abstract=NodeSet, register=False):
    pass


class NodeMapWrapper(Wrapper, abstract=NodeMap, register=False):
    pass


# class NodeTableWrapper(Wrapper, abstract=NodeTable, register=False):
#     pass


class EdgeSetWrapper(Wrapper, abstract=EdgeSet, register=False):
    pass


class EdgeMapWrapper(Wrapper, abstract=EdgeMap, register=False):
    pass


# class EdgeTableWrapper(Wrapper, abstract=EdgeTable, register=False):
#     pass


class GraphWrapper(Wrapper, abstract=Graph, register=False):
    pass


class BipartiteGraphWrapper(Wrapper, abstract=BipartiteGraph, register=False):
    pass


class GraphSageNodeEmbeddingWrapper(
    Wrapper, abstract=GraphSageNodeEmbedding, register=False
):
    pass
