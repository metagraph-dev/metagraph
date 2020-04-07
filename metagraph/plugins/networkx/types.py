from metagraph import ConcreteType, Wrapper, IndexedNodes
from metagraph.types import Graph, DTYPE_CHOICES, WEIGHT_CHOICES
from metagraph.plugins import has_networkx


if has_networkx:
    import networkx as nx

    class NetworkXGraph(Wrapper, abstract=Graph):
        def __init__(
            self,
            nx_graph,
            weight_label=None,
            *,
            weights=None,
            dtype=None,
            node_index=None,
        ):
            self.value = nx_graph
            self.weight_label = weight_label
            self._node_index = node_index
            self._assert_instance(nx_graph, nx.Graph)
            if weight_label is None:
                self._dtype = "bool"
                self._weights = "unweighted"
            else:
                all_values = set()
                for edge in nx_graph.edges(data=True):
                    e_attrs = edge[-1]
                    value = e_attrs[weight_label]
                    all_values.add(value)
                self._dtype = self._determine_dtype(dtype, all_values)
                self._weights = self._determine_weights(weights, all_values)

        def _determine_dtype(self, dtype, all_values):
            if dtype is not None:
                if dtype not in DTYPE_CHOICES:
                    raise ValueError(f"Illegal dtype: {dtype}")
                return dtype

            all_types = {type(v) for v in all_values}
            if not all_types or (all_types - {float, int, bool}):
                return "str"
            for type_ in (float, int, bool):
                if type_ in all_types:
                    return str(type_.__name__)

        def _determine_weights(self, weights, all_values):
            if weights is not None:
                if weights not in WEIGHT_CHOICES:
                    raise ValueError(f"Illegal weights: {weights}")
                return weights

            if self._dtype == "str":
                return "any"
            if self._dtype == "bool":
                if all_values == {True}:
                    return "unweighted"
                return "non-negative"
            else:
                min_val = min(all_values)
                if min_val < 0:
                    return "any"
                elif min_val == 0:
                    return "non-negative"
                else:
                    if self._dtype == "int" and all_values == {1}:
                        return "unweighted"
                    return "positive"

        @property
        def node_index(self):
            if self._node_index is None:
                nodes = tuple(self.value.nodes())
                if type(nodes[0]) == int:
                    nodes = sorted(nodes)
                self._node_index = IndexedNodes(nodes)
            return self._node_index

        @classmethod
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                ret_val.abstract_instance = cls.abstract(
                    is_directed=obj.value.is_directed(),
                    dtype=obj._dtype,
                    weights=obj._weights,
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

    class AutoNetworkXGraphType(ConcreteType, abstract=Graph):
        """
        Interprets a networkx Graph
        If weighted, must use the key "weight"
        """

        value_type = nx.Graph

        @classmethod
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()

                # Determine weights and dtype of graph
                # NOTE: this is expensive and is done for every `get_type` call. To avoid this,
                #       use the NetworkXGraph class instead
                all_values = set()
                all_types = set()
                for edge in obj.edges(data=True):
                    e_attrs = edge[-1]
                    if "weight" not in e_attrs:
                        weights = "unweighted"
                        dtype = "bool"
                        break
                    else:
                        value = e_attrs["weight"]
                        all_values.add(value)
                        all_types.add(type(value))
                else:
                    # Determine dtype and weights using NetworkXGraph objects
                    nx_graph = NetworkXGraph(obj, weight_label="weight")
                    dtype = nx_graph._dtype
                    weights = nx_graph._weights

                ret_val.abstract_instance = cls.abstract(
                    is_directed=False, dtype=dtype, weights=weights
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

    class AutoNetworkXDiGraphType(ConcreteType, abstract=Graph):
        """
        Interprets a networkx DiGraph
        If weighted, must use the key "weight"
        """

        value_type = nx.DiGraph
        abstract_property_specificity_limits = {
            "is_directed": True,
        }

        @classmethod
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()

                # Determine weights and dtype of graph
                # NOTE: this is expensive and is done for every `get_type` call. To avoid this,
                #       use the NetworkXGraph class instead
                all_values = set()
                all_types = set()
                for edge in obj.edges(data=True):
                    e_attrs = edge[-1]
                    if "weight" not in e_attrs:
                        weights = "unweighted"
                        dtype = "bool"
                        break
                    else:
                        value = e_attrs["weight"]
                        all_values.add(value)
                        all_types.add(type(value))
                else:
                    # Determine dtype and weights using NetworkXGraph objects
                    nx_graph = NetworkXGraph(obj, weight_label="weight")
                    dtype = nx_graph._dtype
                    weights = nx_graph._weights

                ret_val.abstract_instance = cls.abstract(
                    is_directed=True, dtype=dtype, weights=weights
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")
