import inspect
from .plugin import (
    AbstractType,
    ConcreteType,
    Wrapper,
    Translator,
    AbstractAlgorithm,
    ConcreteAlgorithm,
)
from collections import defaultdict
from functools import reduce
from typing import Union


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
        ...
        return registry
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
        self.abstract_types = set()
        self.abstract_algorithms = set()
        self.plugin_name_to_concrete_types = defaultdict(set)
        self.plugin_name_to_concrete_algorithms = defaultdict(set)
        self.plugin_name_to_wrappers = defaultdict(set)
        self.plugin_name_to_translators = defaultdict(set)

    @property
    def plugin_names(self):
        return (
            self.plugin_name_to_concrete_types.keys()
            | self.plugin_name_to_concrete_algorithms.keys()
            | self.plugin_name_to_wrappers.keys()
            | self.plugin_name_to_translators.keys()
        )

    @property
    def concrete_types(self):
        return reduce(set.union, self.plugin_name_to_concrete_types.values())

    @property
    def wrappers(self):
        return reduce(set.union, self.plugin_name_to_wrappers.values())

    @property
    def translators(self):
        return reduce(set.union, self.plugin_name_to_translators.values())

    @property
    def concrete_algorithms(self):
        return reduce(set.union, self.plugin_name_to_concrete_algorithms.values())

    def update(self, other_registry) -> None:
        self.abstract_types.update(other_registry.abstract_types)
        self.abstract_algorithms.update(other_registry.abstract_algorithms)
        self.plugin_name_to_concrete_types.update(
            other_registry.plugin_name_to_concrete_types
        )
        self.plugin_name_to_concrete_algorithms.update(
            other_registry.plugin_name_to_concrete_algorithms
        )
        self.plugin_name_to_wrappers.update(other_registry.plugin_name_to_wrappers)
        self.plugin_name_to_translators.update(
            other_registry.plugin_name_to_translators
        )
        return

    def register_abstract(self, obj):
        """
        Decorate abstract classes and functions to include them in the registry
        """
        if isinstance(obj, type):
            if issubclass(obj, AbstractType):
                self.abstract_types.add(obj)
            else:
                raise PluginRegistryError(
                    f"Invalid abstract type for plugin registry: {obj}"
                )
        else:
            if isinstance(obj, AbstractAlgorithm):
                self.abstract_algorithms.add(obj)
            else:
                raise PluginRegistryError(
                    f"Invalid abstract object for plugin registry: {type(obj)}"
                )
        return obj

    def register_concrete(self, plugin_name, obj):
        """
        Decorate concrete classes and functions to include them in the registry
        """
        if isinstance(obj, type):
            if issubclass(obj, ConcreteType):
                self.plugin_name_to_concrete_types[plugin_name].add(obj)
            elif issubclass(obj, Wrapper):
                self.plugin_name_to_wrappers[plugin_name].add(obj)
            else:
                raise PluginRegistryError(
                    f"Invalid concrete type for plugin registry: {obj}"
                )
        else:
            if isinstance(obj, Translator):
                self.plugin_name_to_translators[plugin_name].add(obj)
            elif isinstance(obj, ConcreteAlgorithm):
                self.plugin_name_to_concrete_algorithms[plugin_name].add(obj)
            else:
                raise PluginRegistryError(
                    f"Invalid concrete object for plugin registry: {type(obj)}"
                )
        return obj

    def register_from_modules(
        self, plugin_name: Union[str, None], modules, recurse=True
    ):
        """
        Find and register all suitable objects within modules.

        This only includes objects created within a module or one of its submodules.
        Objects whose names begin with `_` are skipped.

        If ``recurse`` is True, then also recurse into any submodule we find.
        """
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = modules[0]
        for module in modules:
            if not inspect.ismodule(module):
                raise TypeError(
                    f"Expected one or more modules.  Got a type {type(module)} instead."
                )

        # If requested, we could break this out into a function that yields items.
        def _register_module(module, *, recurse, base_name, seen_modules):
            for key, val in vars(module).items():
                if key.startswith("_"):
                    continue
                if isinstance(val, type):
                    val_is_concrete = issubclass(val, (Wrapper, ConcreteType))
                    val_is_abstract = issubclass(val, (AbstractType))
                    if (
                        (val_is_abstract or val_is_concrete)
                        and (
                            val.__module__ == base_name
                            or val.__module__.startswith(base_name + ".")
                        )
                        and val not in {Wrapper, ConcreteType, AbstractType}
                    ):
                        if val_is_abstract:
                            self.register_abstract(val)
                        elif val_is_concrete:
                            if not isinstance(plugin_name, str):
                                raise ValueError(
                                    f"{plugin_name} is not a valid plugin name."
                                )
                            self.register_concrete(plugin_name, val)
                elif isinstance(val, (Translator, ConcreteAlgorithm)):
                    # if val.__wrapped__.__module__.startswith(base_name):  # maybe?
                    if not isinstance(plugin_name, str):
                        raise ValueError(f"{plugin_name} is not a valid plugin name.")
                    self.register_concrete(plugin_name, val)
                elif isinstance(val, (AbstractAlgorithm)):
                    # if val.__wrapped__.__module__.startswith(base_name):  # maybe?
                    self.register_abstract(val)
                elif (
                    recurse
                    and inspect.ismodule(val)
                    and val.__name__.startswith(base_name)
                    and val not in seen_modules
                ):
                    seen_modules.add(val)
                    _register_module(
                        val,
                        recurse=recurse,
                        base_name=base_name,
                        seen_modules=seen_modules,
                    )

        seen_modules = set()
        for module in sorted(modules, key=lambda x: x.__name__.count(".")):
            seen_modules.add(module)
            _register_module(
                module,
                recurse=recurse,
                base_name=module.__name__,
                seen_modules=seen_modules,
            )
