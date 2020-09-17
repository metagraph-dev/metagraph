import operator
import typing
import collections
from functools import reduce
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


def get_plugins(resolver, **kwargs):
    result = OrderedDict()
    plugins = resolver.plugins
    get_name = lambda x: x.__name__
    for plugin_name in sorted(dir(plugins)):
        result[plugin_name] = OrderedDict()
        plugin = getattr(plugins, plugin_name)
        result[plugin_name]["children"] = OrderedDict()
        plugin_part_names = sorted(dir(plugin))
        get_pretty_plugin_part_name = lambda plugin_part_name: " ".join(
            map(str.capitalize, plugin_part_name.split("_"))
        )
        for plugin_part_name in plugin_part_names:
            plugin_part = getattr(plugin, plugin_part_name)
            pretty_plugin_part_name = get_pretty_plugin_part_name(plugin_part_name)
            if plugin_part_name == "abstract_algorithms":

                result[plugin_name]["children"][pretty_plugin_part_name] = {
                    "children": OrderedDict()
                }
                for abstract_algorithm_name in sorted(plugin_part.keys()):
                    result[plugin_name]["children"][pretty_plugin_part_name][
                        "children"
                    ][
                        abstract_algorithm_name
                    ] = {}  # TODO add other useful information here
            elif plugin_part_name == "abstract_types":
                abstract_type_names = sorted(map(get_name, plugin_part))
                result[plugin_name]["children"][pretty_plugin_part_name] = {
                    "children": OrderedDict()
                }
                for abstract_type_name in abstract_type_names:
                    result[plugin_name]["children"][pretty_plugin_part_name][
                        "children"
                    ][
                        abstract_type_name
                    ] = {}  # TODO add other useful information here
            elif plugin_part_name == "concrete_algorithms":
                result[plugin_name]["children"][pretty_plugin_part_name] = {
                    "children": OrderedDict()
                }
                for abstract_algorithm_name in sorted(plugin_part.keys()):
                    concrete_algorithms = sorted(plugin_part[abstract_algorithm_name])
                    result[plugin_name]["children"][pretty_plugin_part_name][
                        "children"
                    ][abstract_algorithm_name] = {"children": OrderedDict()}
                    for concrete_algorithm in concrete_algorithms:
                        concrete_algorithm_name = concrete_algorithm.__name__
                        result[plugin_name]["children"][pretty_plugin_part_name][
                            "children"
                        ][abstract_algorithm_name]["children"][
                            concrete_algorithm_name
                        ] = {}  # TODO add other useful information here
            elif plugin_part_name == "concrete_types":
                concrete_types = sorted(map(get_name, plugin_part))
                result[plugin_name]["children"][pretty_plugin_part_name] = {
                    "children": OrderedDict()
                }
                for concrete_type in concrete_types:
                    result[plugin_name]["children"][pretty_plugin_part_name][
                        "children"
                    ][
                        concrete_type
                    ] = {}  # TODO add other useful information here
            elif plugin_part_name == "translators":
                translator_strings = sorted(
                    [
                        f"{src.__name__} -> {dst.__name__}"
                        for src, dst in plugin_part.keys()
                    ]
                )
                result[plugin_name]["children"][pretty_plugin_part_name] = {
                    "children": OrderedDict()
                }
                for translator_string in translator_strings:
                    result[plugin_name]["children"][pretty_plugin_part_name][
                        "children"
                    ][
                        translator_string
                    ] = {}  # TODO add other useful information here
            elif plugin_part_name == "wrappers":
                abstract_type_names = sorted(dir(plugin_part))
                result[plugin_name]["children"][pretty_plugin_part_name] = {
                    "children": OrderedDict()
                }
                for abstract_type_name in abstract_type_names:
                    abstract_type_namespace = getattr(plugin_part, abstract_type_name)
                    for concrete_type_name in sorted(dir(abstract_type_namespace)):
                        result[plugin_name]["children"][pretty_plugin_part_name][
                            "children"
                        ][
                            concrete_type_name
                        ] = {}  # TODO add other useful information here
            else:
                pass  # TODO consider showing algorithm versions
    return result


def get_abstract_types(resolver, **kwargs):
    return [name for name in list_types(resolver)]


def list_types(resolver, **kwargs):
    """
    Returns an OrderedDict of {abstract_type: [concrete_type, concrete_type, ...]}
    Abstract types and concrete types are sorted alphabetically
    to enable a consistent JSON representation
    """
    t = OrderedDict()

    ats = {at.__name__ for at in resolver.abstract_types}
    cts = resolver.concrete_types

    for at in sorted(ats):
        t[at] = OrderedDict([("type", "abstract_type"), ("children", OrderedDict())])
    for ct in sorted(cts, key=lambda x: x.__name__):
        at = ct.abstract.__name__
        ct_fully_qualified_value_type = (
            f"{ct.value_type.__module__}.{ct.value_type.__qualname__}"
        )
        t[at]["children"][ct.__name__] = OrderedDict(
            [
                ("type", "concrete_type"),
                ("children", {ct_fully_qualified_value_type: {"type": "value_type"}}),
            ]
        )

    return t


def list_translators(resolver, source_type, **kwargs):
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
        resolver.translators, key=lambda x: (x[0].__name__, x[1].__name__)
    ):
        trans = resolver.translators[(src, dst)]
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


def list_algorithms(resolver, **kwargs):
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


def list_algorithm_params(resolver, abstract_pathname: str, **kwargs):
    types = list_types(resolver)
    sig = resolver.abstract_algorithms[abstract_pathname].__signature__

    def resolve_parameter(p) -> OrderedDict:
        result: OrderedDict
        p_class = p if type(p) is type else p.__class__
        if p_class is Combo:
            resolved = [resolve_parameter(psub) for psub in p.types]
            combo_type = " or ".join(r["type"] for r in resolved)
            choices = [c for r in resolved for c in r["choices"]]
            if p.optional:
                combo_type += " or NoneType"  # should this really be type(NoneType)? Python doesn't have a concept of abstract types
                choices.append("NoneType")
            result = OrderedDict([("type", combo_type), ("choices", choices)])
        elif issubclass(p_class, AbstractType):
            choices = list(types[p_class.__name__]["children"].keys())
            result = OrderedDict([("type", p_class.__name__), ("choices", choices)])
        elif getattr(p, "__origin__", None) == collections.abc.Callable:
            result = OrderedDict([("type", p._name), ("choices", [p._name])])
        elif p is typing.Any:
            result = OrderedDict([("type", "Any"), ("choices", ["Any"])])
        else:
            result = OrderedDict(
                [("type", p_class.__name__), ("choices", [p_class.__name__])]
            )
        return result

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


_PRIMITIVE_LIKE_NAME_TO_CLASS = {
    "int": int,
    "float": float,
    "bool": bool,
    "NodeID": int,
    "NoneType": type(None),
}


def _non_concrete_type_to_shell_instance(primitive_like_name):
    if primitive_like_name in _PRIMITIVE_LIKE_NAME_TO_CLASS:
        primitive_like_class = _PRIMITIVE_LIKE_NAME_TO_CLASS[primitive_like_name]
        shell_instance = primitive_like_class.__new__(primitive_like_class)
    elif primitive_like_name == "Callable":
        shell_instance = lambda x: x
    elif primitive_like_name == "Any":
        shell_instance = 1
    else:
        raise ValueError(f"Unhandled type {primitive_like_name}")
    return shell_instance


def solve_algorithm(
    resolver, abstract_pathname: str, params_description: dict, **kwargs
):
    """
    abstract_pathname: string with dotted path and abstract function name
    params_description: dict like {'x': {'abstract_type': 'NodeMap', 'concrete_type': 'NumpyNodeMapType'}, ...}
        - keys are parameter names, e.g. 'x'
        - values are dicts with keys 'abstract_type' and 'concrete_type'
    """
    if abstract_pathname not in resolver.abstract_algorithms:
        raise ValueError(f'No abstract algorithm "{abstract_pathname}" exists')

    params = {}
    for pname, pdescription in params_description.items():
        concrete_type_name = pdescription["concrete_type"]
        concrete_type = None
        abstract_type_names = pdescription["abstract_type"].split(" or ")
        for abstract_type_name in abstract_type_names:
            abstract_type_namespace = getattr(resolver.types, abstract_type_name, None)
            if abstract_type_namespace is not None:
                concrete_type = getattr(
                    abstract_type_namespace, concrete_type_name, None
                )
                break
        if concrete_type is None:
            params[pname] = _non_concrete_type_to_shell_instance(abstract_type_name)
            continue
        concrete_type_value_type = concrete_type.value_type
        # Convert params from classes to shells of instances
        # (needed by code which expects instances)
        params[pname] = concrete_type_value_type.__new__(concrete_type_value_type)

    solutions = {}
    concrete_algorithms = resolver.concrete_algorithms.get(abstract_pathname, {})
    plans = [AlgorithmPlan.build(resolver, ca, **params) for ca in concrete_algorithms]

    for plan_index, plan in enumerate(plans):
        if not plan.unsatisfiable:
            # TODO store backpointers in the resolver instead of doing an O(n) lookup here
            for plugin_name in dir(resolver.plugins):
                plugin = getattr(resolver.plugins, plugin_name)
                if abstract_pathname in plugin.concrete_algorithms:
                    plugin_concrete_algorithms = plugin.concrete_algorithms[
                        abstract_pathname
                    ]
                    if plan.algo in plugin_concrete_algorithms:
                        parameter_data = OrderedDict()
                        for parameter in plan.algo.__signature__.parameters.values():
                            translation_path_dict = OrderedDict(
                                [("translation_path", {"type": "translation_path"}),]
                            )
                            # TODO abstract common functionality from here and AlgorithmPlan.__repr__ into a class method for AlgorithmPlan
                            if parameter.name in plan.required_translations:
                                mst = plan.required_translations[parameter.name]
                                translation_types = [mst.src_type] + mst.dst_types
                                translation_path_dict["translation_path"][
                                    "children"
                                ] = OrderedDict()
                                for ct in translation_types:
                                    translation_path_dict["translation_path"][
                                        "children"
                                    ][ct.__name__] = {
                                        "type": "translation_path_element"
                                    }
                                translation_path_dict["translation_path"][
                                    "translation_path_length"
                                ] = len(translation_types)
                            else:
                                translation_path_dict["translation_path"][
                                    "translation_path_length"
                                ] = 0

                            # TODO make metagraph types have a __qualname__
                            annotation_string = getattr(
                                parameter.annotation, "__qualname__", None
                            )
                            if annotation_string is None:
                                annotation_string = getattr(
                                    parameter.annotation,
                                    "__name__",
                                    str(parameter.annotation),
                                )

                            parameter_data[parameter.name] = OrderedDict(
                                [
                                    ("type", "parameter"),
                                    ("annotation", annotation_string),
                                    ("children", translation_path_dict),
                                ]
                            )
                        # TODO make this an ordered dict
                        solutions[f"plan_{plan_index}"] = {
                            "type": "plan",
                            "plan_index": plan_index,
                            "children": {
                                f"Algorithm Name: {plan.algo.func.__name__}": {},
                                f"Plugin: {plugin_name}": {},
                                "Params": {"children": parameter_data},
                                f"Return Type: {plan.algo.__signature__.return_annotation}": {},
                            },
                        }
                        break
        return solutions
