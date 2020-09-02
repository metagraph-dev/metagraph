from collections import OrderedDict
from ..core.plugin import AbstractType, ConcreteType, ConcreteAlgorithm
from ..core.typing import Combo
from ..core.planning import MultiStepTranslator, AlgorithmPlan

# Guiding principles
# 1. Make API functions testable in Python
# 2. API functions should return Python objects which are easily converted into JSON (dict, list, str, int, bool)
# 3. Make the service handle all conversion to/from JSON
# 4. Make object structure as consistent as possible so decoding on Javascript side is easier


def normalize_abstract_type(resolver, abstract, **kwargs):
    if type(abstract) is type and issubclass(abstract, AbstractType):
        type_class = abstract
        abstract = abstract.__name__
    else:
        for klass in resolver.abstract_types:
            if klass.__name__ == abstract:
                type_class = klass
                break
        else:
            raise ValueError(f"Unknown abstract type: {abstract}")
    return (abstract, type_class)


def normalize_concrete_type(resolver, abstract, concrete, **kwargs):
    abstract, abstract_class = normalize_abstract_type(resolver, abstract)

    if type(concrete) is type and issubclass(concrete, ConcreteType):
        if concrete.abstract is not abstract_class:
            raise ValueError(
                f"Mismatch in abstract type provided and abstract type of "
                f"concrete provided: {abstract} vs {concrete.abstract.__name__}"
            )
        type_class = concrete
        concrete = concrete.__name__
    else:
        for klass in resolver.concrete_types:
            if klass.__name__ == concrete and klass.abstract.__name__ == abstract:
                type_class = klass
                break
        else:
            raise ValueError(f"Unknown concrete type: {abstract}/{concrete}")
    return (concrete, type_class)


def _dwim_plugins_dict_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        dwimmed_value = value
    elif isinstance(value, (list, tuple, set)):
        dwimmed_value = [_dwim_plugins_dict_value(sub_element) for sub_element in value]
    elif isinstance(value, dict):
        dwimmed_value = _dwim_plugins_dict(value)
    else:
        dwimmed_value = str(value)
    return dwimmed_value


def _dwim_plugins_dict_key(key):
    if (
        isinstance(key, tuple)
        and len(key) == 2
        and all(issubclass(element, ConcreteType) for element in key)
    ):
        src, dst = key
        dwimmed_key = f"{src.__name__} -> {dst.__name__}"
    else:
        dwimmed_key = key
    return dwimmed_key


def _dwim_plugins_dict(plugins_dict: dict) -> dict:
    """Modifies content in place"""
    plugins_dict = {
        _dwim_plugins_dict_key(key): _dwim_plugins_dict_value(value)
        for key, value in plugins_dict.items()
    }
    return plugins_dict


def get_plugins(resolver, **kwargs):
    return _dwim_plugins_dict(resolver.plugins.to_dict())


def get_abstract_types(resolver, **kwargs):
    return [name for name in list_types(resolver)]


def list_types(resolver, filters=None, **kwargs):
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


def list_translators(resolver, source_type, filters=None, **kwargs):
    if filters and "plugin" in filters:
        plugin = getattr(resolver.plugins, filters["plugin"])
        filtered_translators = plugin.translators
        plugins = [filters["plugin"]]
    else:
        filtered_translators = resolver.translators
        plugins = sorted(dir(resolver.plugins))

    source_type, source_class = normalize_abstract_type(resolver, source_type)

    types = list_types(resolver)
    primary_types = types[source_type]["children"].copy()
    secondary_types = OrderedDict()
    for at in source_class.unambiguous_subcomponents:
        for ct_name, ct in types[at.__name__]["children"].items():
            secondary_types[ct_name] = ct
    primary_translators = OrderedDict()
    secondary_translators = OrderedDict()
    for src, dst in sorted(
        filtered_translators, key=lambda x: (x[0].__name__, x[1].__name__)
    ):
        trans = filtered_translators[(src, dst)]
        # Find which plugin the translator came from
        for plugin in plugins:
            trans_keys = getattr(resolver.plugins, plugin).translators
            if (src, dst) in trans_keys:
                break
        else:
            plugin = "Unknown"
        trans_info = [
            ("type", "translator"),
            ("name", trans.func.__name__),
            ("plugin", plugin),
            ("module", trans.func.__module__),
        ]

        src_at = src.abstract.__name__
        if src_at == source_type:
            primary_translators[f"{src.__name__} -> {dst.__name__}"] = OrderedDict(
                trans_info
            )
        elif src.__name__ in secondary_types and dst.__name__ in secondary_types:
            secondary_translators[f"{src.__name__} -> {dst.__name__}"] = OrderedDict(
                trans_info
            )

    return {
        "primary_types": primary_types,
        "secondary_types": secondary_types,
        "primary_translators": primary_translators,
        "secondary_translators": secondary_translators,
    }


def list_algorithms(resolver, filters=None, **kwargs):
    if filters and "plugin" in filters:
        plugin = getattr(resolver.plugins, filters["plugin"])
        # Include abstract algos defined in plugin along with abstract algos matching concrete algos defined
        abstract_algorithms = set(plugin.abstract_algorithms)
        for aa in plugin.concrete_algorithms:
            abstract_algorithms.add(aa)
        plugins = [filters["plugin"]]
    else:
        abstract_algorithms = resolver.abstract_algorithms.keys()
        plugins = sorted(dir(resolver.plugins))

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


def list_algorithm_params(resolver, abstract_pathname, **kwargs):
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


# Translator object will contain:
# - src_type: str
# - dst_type: str
# - result_type: str [multi-step, direct, unsatisfiable, null]
# - solution: list of str (will be empty list for unsatisfiable)


def solve_translator(
    resolver, src_abstract, src_concrete, dst_abstract, dst_concrete, **kwargs
):
    src_type, src_class = normalize_concrete_type(resolver, src_abstract, src_concrete)
    dst_type, dst_class = normalize_concrete_type(resolver, dst_abstract, dst_concrete)

    mst = MultiStepTranslator.find_translation(resolver, src_class, dst_class)
    if mst.unsatisfiable:
        result_type = "unsatisfiable"
    elif len(mst) == 0:
        result_type = "null"
    elif len(mst) > 1:
        result_type = "multi-step"
    else:
        result_type = "direct"

    return {
        "src_type": src_type,
        "dst_type": dst_type,
        "result_type": result_type,
        "solution": [mst.src_type.__name__] + [step.__name__ for step in mst.dst_types],
    }


def solve_algorithm(resolver, abstract_pathname, params, returns, **kwargs):
    """
    abstract_pathname: string with dotted path and abstract function name
    params: dict of parameter name to type (class or ConcreteType)
    returns: list of types (class or ConcreteType)
    """
    if abstract_pathname not in resolver.abstract_algorithms:
        raise ValueError(f'No abstract algorithm "{abstract_pathname}" exists')

    # Convert params from classes to shells of instances
    # (needed by code which expects instances)
    params = {pname: p.__new__(p) for pname, p in params.items()}

    solutions = []
    for ca_name, ca in resolver.concrete_algorithms.get(abstract_pathname, {}).items():
        plan = AlgorithmPlan.build(resolver, ca_name, **params)
        solutions.append(
            {
                "algo_name": ca.func.__name__,
                "plugin": None,
                "module": ca.func.__module__,
                "unsatisfiable": False,
                "params": None,
                "returns": None,
            }
        )
