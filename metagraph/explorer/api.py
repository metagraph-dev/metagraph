import json
import asyncio
from collections import OrderedDict
from ..core.plugin import AbstractType, ConcreteType

# Guiding principles
# 1. Make API functions testable in Python
# 2. API functions should return Python objects which are easily converted into JSON (dict, list, str, int, bool)
# 3. Make the service handle all conversion to/from JSON
# 4. Make object structure as consistent as possible so decoding on Javascript side is easier


def list_plugins(resolver):
    # This is tricky because we can find the plugins for the default resolver, but not for a custom resolver
    raise NotImplementedError()


def get_abstract_types(resolver):
    return [name for name in list_types(resolver)]


def list_types(resolver, filters=None):
    """
    Returns an OrderedDict of {abstract_type: [concrete_type, concrete_type, ...]}
    Abstract types and concrete types are sorted alphabetically
    to enable a consistent JSON representation
    """
    if filters:
        raise NotImplementedError()

    t = OrderedDict()
    ats = {at.__name__ for at in resolver.abstract_types}
    for at in sorted(ats):
        t[at] = OrderedDict([("type", "abstract_type"), ("children", OrderedDict())])
    for ct in sorted(resolver.concrete_types, key=lambda x: x.__name__):
        at = ct.abstract.__name__
        t[at]["children"][ct.__name__] = OrderedDict([("type", "concrete_type")])
    return t


def list_translators(resolver, source_type, filters=None):
    if filters:
        raise NotImplementedError()

    # Normalize source_types
    if type(source_type) is type and issubclass(source_type, AbstractType):
        source_class = source_type
        source_type = source_type.__name__
    else:
        for klass in resolver.abstract_types:
            if klass.__name__ == source_type:
                source_class = klass
                break
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

    types = list_types(resolver, filters=filters)
    primary_types = list(types[source_type]["children"].keys())
    secondary_types = [
        ct
        for at in source_class.unambiguous_subcomponents
        for ct in types[at.__name__]["children"]
    ]
    secondary_types.sort()
    translators = []
    for src, dst in resolver.translators.keys():
        at = src.abstract.__name__
        if at == source_type:
            translators.append([src.__name__, dst.__name__])
    translators.sort()

    return {
        "primary_types": primary_types,
        "secondary_types": secondary_types,
        "translators": translators,
    }


def list_algorithms(resolver, filters=None):
    if filters:
        raise NotImplementedError()

    d = OrderedDict()
    for aa in sorted(resolver.abstract_algorithms.keys()):
        root = d
        *paths, algo = aa.split(".")
        for path in paths:
            if path not in root:
                root[path] = OrderedDict(
                    [("type", "path"), ("children", OrderedDict())]
                )
            assert root[path]["type"] == "path"
            root = root[path]["children"]
        root[algo] = OrderedDict(
            [
                ("type", "abstract_algorithm"),
                ("full_path", aa),
                ("children", OrderedDict()),
            ]
        )
        concretes = root[algo]["children"]
        # ConcreteAlgorithms don't have a guaranteed unique name, so use the plugin name as a surrogate
        for plugin in sorted(dir(resolver.plugins)):
            plug = getattr(resolver.plugins, plugin)
            for aa_name, ca_set in plug.concrete_algorithms.items():
                if aa_name == aa:
                    concretes[plugin] = OrderedDict([("type", "concrete_algorithm")])
                    ca = list(ca_set)[0]  # ca_set is guaranteed to be len 1
                    if hasattr(ca.func, "__name__"):
                        concretes[plugin]["name"] = ca.func.__name__
                    if hasattr(ca.func, "__module__"):
                        concretes[plugin]["module"] = ca.func.__module__
    return d


def list_algorithm_params(resolver, abstract_pathname):
    raise NotImplementedError()

    sig = resolver.abstract_algorithms[abstract_pathname].__signature__
    params = OrderedDict()
    for pname, p in sig.parameters.items():
        pass
    returns = []
    if getattr(sig.return_annotation, "__origin__", None) is tuple:
        for ret in sig.return_annotation.__args__:
            pass
    else:
        pass
    return {
        "parameters": params,
        "returns": returns,
    }


def solve_translator(resolver, src_type, dst_type):
    raise NotImplementedError()


def solve_algorithm(resolver, abstract_pathname, params, returns):
    raise NotImplementedError()
