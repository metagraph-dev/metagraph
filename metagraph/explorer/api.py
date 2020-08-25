from collections import OrderedDict
from ..core.plugin import AbstractType, ConcreteType
from ..core.typing import Combo

# Guiding principles
# 1. Make API functions testable in Python
# 2. API functions should return Python objects which are easily converted into JSON (dict, list, str, int, bool)
# 3. Make the service handle all conversion to/from JSON
# 4. Make object structure as consistent as possible so decoding on Javascript side is easier


def list_plugins(resolver):
    # TODO: change to a dict with the module listed as well as the plugin name
    return list(sorted(dir(resolver.plugins)))


def get_abstract_types(resolver):
    return [name for name in list_types(resolver)]


def list_types(resolver, filters=None):
    """
    Returns an OrderedDict of {abstract_type: [concrete_type, concrete_type, ...]}
    Abstract types and concrete types are sorted alphabetically
    to enable a consistent JSON representation
    """
    t = OrderedDict()

    if filters and "plugin" in filters:
        plugin = getattr(resolver.plugins, filters["plugin"])
        # Include abstract types defined in plugin along with abstract types matching concrete types defined
        ats = {at.__name__ for at in plugin.abstract_types}
        for ct in plugin.concrete_types:
            at = ct.abstract.__name__
            ats.add(at)
        cts = plugin.concrete_types
    else:
        ats = {at.__name__ for at in resolver.abstract_types}
        cts = resolver.concrete_types

    for at in sorted(ats):
        t[at] = OrderedDict([("type", "abstract_type"), ("children", OrderedDict())])
    for ct in sorted(cts, key=lambda x: x.__name__):
        at = ct.abstract.__name__
        t[at]["children"][ct.__name__] = OrderedDict([("type", "concrete_type")])

    return t


def list_translators(resolver, source_type, filters=None):
    if filters and "plugin" in filters:
        plugin = getattr(resolver.plugins, filters["plugin"])
        filtered_translators = plugin.translators
    else:
        filtered_translators = resolver.translators

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

    types = list_types(resolver)
    primary_types = list(types[source_type]["children"].keys())
    secondary_types = [
        ct
        for at in source_class.unambiguous_subcomponents
        for ct in types[at.__name__]["children"]
    ]
    secondary_types.sort()
    translators = []
    for src, dst in filtered_translators.keys():
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
    if filters and "plugin" in filters:
        plugin = getattr(resolver.plugins, filters["plugin"])
        # Include abstract algos defined in plugin along with abstract algos matching concrete algos defined
        abstract_algorithms = set(plugin.abstract_algorithms)
        for aa in plugin.concrete_algorithms:
            abstract_algorithms.add(aa)
        plugins = [filters["plugin"]]
    else:
        abstract_algorithms = resolver.abstract_algorithms.keys()
        plugins = list_plugins(resolver)

    d = OrderedDict()
    for aa in sorted(abstract_algorithms):
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
        for plugin in plugins:
            plug = getattr(resolver.plugins, plugin)
            for aa_name, ca_set in plug.concrete_algorithms.items():
                if aa_name == aa:
                    ca = list(ca_set)[0]  # ca_set is guaranteed to be len 1
                    funcname = ca.func.__name__
                    cname = f"[{plugin}] {funcname}"
                    concretes[cname] = OrderedDict(
                        [
                            ("type", "concrete_algorithm"),
                            ("name", funcname),
                            ("plugin", plugin),
                        ]
                    )
                    if hasattr(ca.func, "__module__"):
                        concretes[cname]["module"] = ca.func.__module__
    return d


def list_algorithm_params(resolver, abstract_pathname):
    types = list_types(resolver)
    sig = resolver.abstract_algorithms[abstract_pathname].__signature__

    def resolve_parameter(p):
        p_class = p if type(p) is type else p.__class__
        if p_class is Combo:
            resolved = [resolve_parameter(psub) for psub in p.types]
            combo_type = " or ".join(r["type"] for r in resolved)
            choices = [c for r in resolved for c in r["choices"]]
            return OrderedDict([("type", combo_type), ("choices", choices)])
        elif issubclass(p_class, AbstractType):
            choices = list(types[p_class.__name__]["children"].keys())
            return OrderedDict([("type", p_class.__name__), ("choices", choices)])
        else:
            return OrderedDict([("type", p_class.__name__)])

    params = OrderedDict()
    for pname, p in sig.parameters.items():
        params[pname] = resolve_parameter(p.annotation)

    returns = []
    if getattr(sig.return_annotation, "__origin__", None) is tuple:
        for ret in sig.return_annotation.__args__:
            returns.append(resolve_parameter(ret))
    else:
        returns.append(resolve_parameter(sig.return_annotation))
    return {
        "parameters": params,
        "returns": returns,
    }


def solve_translator(resolver, src_type, dst_type):
    raise NotImplementedError()


def solve_algorithm(resolver, abstract_pathname, params, returns):
    raise NotImplementedError()
