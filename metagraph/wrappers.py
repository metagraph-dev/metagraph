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
    NodeEmbedding,
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


class NodeEmbeddingWrapper(Wrapper, abstract=NodeEmbedding, register=False):
    def __init__(self, matrix, nodes=None):
        super().__init__()
        self._assert(
            type(matrix).Type.compute_abstract_properties(matrix, {"is_dense"})[
                "is_dense"
            ],
            f"Matrix {matrix} must be dense",
        )
        self.matrix = matrix
        if nodes is not None:
            nodes_dtype = type(nodes).Type.compute_abstract_properties(
                nodes, {"dtype"}
            )["dtype"]
            self._assert(
                nodes_dtype == "int",
                f"Node map {nodes} must have dtype of int rather than {nodes_dtype}",
            )
        self.nodes = nodes

    class TypeMixin:
        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: Set[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"matrix_dtype"} - ret.keys():
                if prop == "matrix_dtype":
                    ret[prop] = type(obj.matrix).Type.compute_abstract_properties(
                        obj.matrix, {"dtype"}, {}
                    )

            return ret

        @classmethod
        def assert_equal(
            cls,
            obj1,
            obj2,
            aprops1,
            aprops2,
            cprops1,
            cprops2,
            *,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"

            matrix_class = type(obj1.matrix).Type
            matrix_class.assert_equal(
                obj1.matrix,
                obj2.matrix,
                {"dtype": aprops1["matrix_dtype"]},
                {"dtype": aprops2["matrix_dtype"]},
                {},
                {},
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )

            nodes_class = type(obj1.nodes).Type
            nodes_class.assert_equal(
                obj1.nodes,
                obj2.nodes,
                {},
                {},
                {},
                {},
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )


class GraphSageNodeEmbeddingWrapper(
    Wrapper, abstract=GraphSageNodeEmbedding, register=False
):
    pass
