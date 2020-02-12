"""
Use this module to create a metagraph plugin

Use as follows:
mgp = metagraph.plugin_maker.PluginMaker('foo')

# Create a new abstract type
class MyCustomAbstractType(mgp.AbstractType):
    name = 'my_custom_type'

# Create a new concrete type around your wrapper
class MyWrapper:
    def __init__(self, obj, flag=True):
        self.obj = obj
        self.flag = flag

class MyCustomConcreteType(mgp.ConcreteType):
    abstract = MyCustomAbstractType
    value_class = MyWrapper

# Create a translator
@mgp.translator
def nx_to_mycustom(x: NetworkXGraphType, **props) -> MyCustomConcreteType:
    return MyWrapper.from_nx(x)

# Create an abstract algorithm
@mgp.abstract_algorithm('link_analysis.CustomPageRank')
def custpr(g: Graph) -> Vector:
    pass

# Create a concrete algorithm
@mgp.concrete_algorithm('link_analysis.CustomPageRank')
def mycustpr(g: MyCustomConcreteType) -> NumpyVector:
    result = ...
    return results
"""
from .plugin import (
    AbstractType,
    ConcreteType,
    Translator,
    AbstractAlgorithm,
    ConcreteAlgorithm,
)


def normalize_type(typ, klass, varname):
    # Scalars are okay
    if typ in (bool, int, float, str):
        return typ
    # ForwardRef is just the string
    if hasattr(typ, "__forward_arg__"):
        typ = typ.__forward_arg__
    elif isinstance(typ, klass):
        typ = klass.name
    if not isinstance(typ, str):
        raise PluginRegistryError(f'Illegal type for "{varname}": {type(typ)}')
    return typ


class PluginRegistryError(Exception):
    pass


class PluginRegistry:
    """
    PluginRegistry for use by libraries implementing new types, translators, and algorithms for metagraph.

    Usage:
     - Create a plugin registry for your library
       plugreg = PluginRegistry('my_library')
       # Note: This will either create or retrieve an existing plugin registry for 'my_library', allowing
       #       each module to "re-create" the registry, but in reality they just get a handle to the same
       #       registry object, similar to how the logging module works.
     - Create subclasses of AbstractType, ConcreteType, and decorate translators, abstract_algorithms,
       and concrete_algorithms.
     - Wrap each of these in @plugreg.register to register them with the system
     - Create a master plugin.py
       - Create a `plugreg = PluginRegistry('my_library')` to get a handle on the shared registry object
       - Import all modules which register objects. Until they are imported, they haven't actually been
         registered with the system. Make sure any nested modules are imported as well so the registery is
         fully populated.
     - Create entry_points.txt
       [metagraph.plugins]
       registry = plugin:plugreg
       # This points to plugin.py, and calls the plugreg object. The shared registry object (plugreg)
       # is designed to give entry_points a complete list of all registered objects for 'my_library'
    """

    _registry = {}

    def __init__(self, name):
        self.name = name
        if name not in PluginRegistry._registry:
            PluginRegistry._registry[name] = {
                "abstract_types": [],
                "concrete_types": [],
                "translators": [],
                "abstract_algorithms": [],
                "concrete_algorithms": [],
            }

    def register(self, obj):
        """
        Decorate classes and functions to include them in the registry
        """
        if isinstance(obj, AbstractType):
            PluginRegistry._registry[self.name]["abstract_types"].append(obj)
        elif isinstance(obj, ConcreteType):
            PluginRegistry._registry[self.name]["concrete_types"].append(obj)
        elif isinstance(obj, Translator):
            PluginRegistry._registry[self.name]["translators"].append(obj)
        elif isinstance(obj, AbstractAlgorithm):
            PluginRegistry._registry[self.name]["abstract_algorithms"].append(obj)
        elif isinstance(obj, ConcreteAlgorithm):
            PluginRegistry._registry[self.name]["concrete_algorithms"].append(obj)
        else:
            raise PluginRegistryError(f"Invalid item for plugin registry: {type(obj)}")

        return obj

    # entry_points wants a callable, so we provide that functionality here
    def __call__(self):
        return PluginRegistry._registry[self.name]


# class AbstractType:
#     _registry = {}
#     name = None
#
#     def __init_subclass__(cls):
#         if cls.name is None:
#             raise PluginMakerError('AbstractType must define "name" class attribute')
#         if cls.name in cls._registry:
#             raise PluginMakerError(f'"{cls.name}" is already a registered AbstractType name')
#         cls._registry[cls.name] = cls
#
# class ConcreteType:
#     _registry = {}
#     abstract = None
#     name = None
#
#     def __init_subclass__(cls):
#         if cls.name is None:
#             raise PluginMakerError('ConcreteType must define "name" class attribute')
#         if cls.abstract is None:
#             raise PluginMakerError('ConcreteType must define "abstract" class attribute')
#         cls.abstract = normalize_type(cls.abstract, AbstractType, 'abstract')
#         cls._registry[cls.name] = cls
#
# class Translator:
#     _registry = {}
#
#     @classmethod
#     def parse_sig_decorator(cls, func):
#         hints = func.__annotations__
#         if 'return' not in hints:
#             raise PluginMakerError('Must include return type in signature')
#         ret = hints.pop('return')
#         if len(hints) != 1:
#             raise PluginMakerError('Translator signature must be `func(src: Class1, **kwargs) -> Class2`')
#         src = list(hints.values())[0]
#         src = normalize_type(src, ConcreteType, 'src')
#         ret = normalize_type(ret, ConcreteType, 'return')
#         cls._registry[(src, ret)] = func
#
# class AbstractAlgorithmDecorator:
#     _registry = {}
#
#     def __init__(self, algo_path):
#         self.algo_path = algo_path
#
#     def __call__(self, func):
#         hints = func.__annotations__
#         argnames = list(func.__code__.co_varnames) + ['return']
#         # Verify all concrete_types are abstract concrete_types
#         for argname in argnames:
#             item = hints[argname]
#             if hasattr(item, '__origin__') and hasattr(item, '__args__'):
#                 tmp = []
#                 for i, it in enumerate(item.__args__):
#                     it = normalize_type(it, AbstractType, f'{argname}[{i}]')
#                     tmp.append(it)
#                 item.__args__ = tuple(tmp)
#             else:
#                 item = normalize_type(item, AbstractType, argname)
#                 hints[argname] = item
#
#         self._registry[self.algo_path] = (func, hints)
#
# class ConcreteAlgorithmDecorator:
#     _registry = {}
#
#     def __init__(self, algo_path):
#         self.algo_path = algo_path
#
#     def __call__(self, func):
#         hints = func.__annotations__
#         argnames = list(func.__code__.co_varnames) + ['return']
#         # Verify all concrete_types are concrete concrete_types
#         for argname in argnames:
#             item = hints[argname]
#             if hasattr(item, '__origin__') and hasattr(item, '__args__'):
#                 tmp = []
#                 for i, it in enumerate(item.__args__):
#                     it = normalize_type(it, ConcreteType, f'{argname}[{i}]')
#                     tmp.append(it)
#                 item.__args__ = tuple(tmp)
#             else:
#                 item = normalize_type(item, ConcreteType, argname)
#                 hints[argname] = item
#
#         self._registry[self.algo_path] = (func, hints)
#
# self.AbstractType = AbstractType
# self.ConcreteType = ConcreteType
# self.translator = Translator.parse_sig_decorator
# self.abstract_algorithm = AbstractAlgorithmDecorator
# self.concrete_algorithm = ConcreteAlgorithmDecorator


del normalize_type
