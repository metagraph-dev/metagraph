from .plugin import (
    AbstractType,
    ConcreteType,
    Translator,
    AbstractAlgorithm,
    ConcreteAlgorithm,
)


class PluginRegistryError(Exception):
    pass


class PluginRegistry:
    """
    PluginRegistry for use by libraries implementing new types, translators, and algorithms for metagraph.

    Example Usage
    -------------
    # /plugins.py
    registry = metagraph.plugin_registry.PluginRegistry()
    def find_plugins():
        # Import modules here to ensure they are registered, but avoid circular imports
        from . import my_types, my_translators, my_algorithms
        return registry.plugins

    # Add entry_points to setup.py
    setup(
        ...
        entry_points={"metagraph.plugins": "plugins = plugins:find_plugins"},
        ...
    )

    # /my_types.py
    from .plugins import registry
    from metagraph import AbstractType, Wrapper

    @registry.register
    class MyCustomType(AbstractType):
        pass

    @registry.register
    class MyWrapper(Wrapper, abstract=MyCustomType):
        allowed_props = {'flag'}

        def __init__(self, value, flag=True):
            self.value = value
            self.flag = flag

    # /my_translators.py
    from .plugins import registry
    from .my_types import MyWrapper
    from metagraph import translator
    import networkx as nx

    @translator(registry=registry)
    def nx_to_mycustom(x: nx.Graph, **props) -> MyWrapper:
        # modify x
        return MyWrapper(x, flag=False)

    # /my_algorithms.py
    from .plugins import registry
    from .my_types import MyWrapper
    from metagraph import abstract_algorithm, concrete_algorithm
    import metagraph as mg

    # Create an abstract algorithm
    @abstract_algorithm('link_analysis.CustomPageRank', registry=registry)
    def custpr(g: Graph) -> Vector:
        pass

    @concrete_algorithm('link_analysis.CustomPageRank', registry=registry)
    def mycustpr(g: MyWrapper) -> mg.types.NumpyVector:
        result = ...
        return results
    """

    def __init__(self):
        self.plugins = {
            "abstract_types": [],
            "wrappers": [],
            "translators": [],
            "abstract_algorithms": [],
            "concrete_algorithms": [],
        }

    def register(self, obj):
        """
        Decorate classes and functions to include them in the registry
        """
        if isinstance(obj, AbstractType):
            self.plugins["abstract_types"].append(obj)
        elif isinstance(obj, ConcreteType):
            self.plugins["wrappers"].append(obj)
        elif isinstance(obj, Translator):
            self.plugins["translators"].append(obj)
        elif isinstance(obj, AbstractAlgorithm):
            self.plugins["abstract_algorithms"].append(obj)
        elif isinstance(obj, ConcreteAlgorithm):
            self.plugins["concrete_algorithms"].append(obj)
        else:
            raise PluginRegistryError(
                f"Invalid object for plugin registry: {type(obj)}"
            )

        return obj
