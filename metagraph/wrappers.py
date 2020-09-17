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


class CompositeGraphWrapper(GraphWrapper, abstract=Graph, register=False):
    def __init__(self, edges, nodes=None):
        super().__init__()
        self.edges = edges
        self.nodes = nodes

    class TypeMixin:
        _edge_prop_map = {
            "is_directed": "is_directed",
            "edge_dtype": "dtype",
            "edge_has_negative_weights": "has_negative_weights",
        }
        _node_prop_map = {
            "node_dtype": "dtype",
        }

        @classmethod
        def _extract_props(cls, props, map):
            ret = {}
            for gprop, prop in map.items():
                if gprop in props:
                    ret[prop] = props[gprop]
            return ret

        @classmethod
        def _compute_subprops(cls, ret, obj, props, map):
            gprops_needed = (map.keys() & props) - ret.keys()
            if gprops_needed and obj is not None:
                klass = type(obj)
                vals = klass.Type.compute_abstract_properties(
                    obj, {prop for gprop, prop in map.items() if gprop in gprops_needed}
                )
                for gprop, prop in map.items():
                    if prop in vals:
                        ret[gprop] = vals[prop]

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: Set[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"node_type", "edge_type"} - ret.keys():
                if prop == "node_type":
                    if obj.nodes is None or isinstance(obj.nodes, NodeSetWrapper):
                        ret["node_type"] = "set"
                        ret["node_dtype"] = None
                    elif isinstance(obj.nodes, NodeMapWrapper):
                        ret["node_type"] = "map"
                    # elif isinstance(obj.nodes, NodeTableWrapper):
                    #     ret["node_type"] = "table"
                if prop == "edge_type":
                    if isinstance(obj.edges, EdgeSetWrapper):
                        ret["edge_type"] = "set"
                        ret["edge_dtype"] = None
                        ret["edge_has_negative_weights"] = None
                    elif isinstance(obj.edges, EdgeMapWrapper):
                        ret["edge_type"] = "map"
                    # elif isinstance(obj.edges, EdgeTableWrapper):
                    #     ret["edge_type"] = "table"

            cls._compute_subprops(ret, obj.edges, props, cls._edge_prop_map)
            cls._compute_subprops(ret, obj.nodes, props, cls._node_prop_map)
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

            edge_class = type(obj1.edges).Type
            edge_class.assert_equal(
                obj1.edges,
                obj2.edges,
                cls._extract_props(aprops1, cls._edge_prop_map),
                cls._extract_props(aprops2, cls._edge_prop_map),
                {},
                {},
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            if aprops1["node_type"] != "set":
                node_class = type(obj1.nodes).Type
                node_class.assert_equal(
                    obj1.nodes,
                    obj2.nodes,
                    cls._extract_props(aprops1, cls._node_prop_map),
                    cls._extract_props(aprops2, cls._node_prop_map),
                    {},
                    {},
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )


class BipartiteGraphWrapper(Wrapper, abstract=BipartiteGraph, register=False):
    pass
