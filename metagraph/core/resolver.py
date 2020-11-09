"""A Resolver manages a collection of plugins, resolves types, and dispatches
to concrete algorithms.

"""
import copy
import inspect
import warnings
from collections import defaultdict, abc
from typing import (
    List,
    Tuple,
    Set,
    Dict,
    DefaultDict,
    Callable,
    Optional,
    Any,
    Union,
    TypeVar,
)
from .plugin import (
    AbstractType,
    ConcreteType,
    Wrapper,
    Translator,
    AbstractAlgorithm,
    ConcreteAlgorithm,
)
from .planning import MultiStepTranslator, AlgorithmPlan, TranslationMatrix
from .entrypoints import load_plugins
from . import typing as mgtyping
from .. import config
from ..types import NodeID
import numpy as np


class NamespaceError(Exception):
    pass


class AlgorithmWarning(Warning):
    pass


class Namespace:
    """Helper class to construct arbitrary nested namespaces of objects on the fly.

    Objects are registered with their full dotted attribute path, and the appropriate
    nested namespace object structure is automatically constructed as needed.  There
    is no removal mechanism.
    """

    def __init__(self):
        self._registered = set()

    def _register(self, path: str, obj):
        parts = path.split(".")
        name = parts[0]
        self._registered.add(name)
        if len(parts) == 1:
            if hasattr(self, name):
                raise NamespaceError(f"Name already registered: {name}")
            setattr(self, name, obj)
        else:
            if not hasattr(self, name):
                setattr(self, name, Namespace())
            getattr(self, name)._register(".".join(parts[1:]), obj)

    def __dir__(self):
        return self._registered

    def to_dict(self):
        result = {}
        for key in self._registered:
            value = getattr(self, key)
            if isinstance(value, Namespace):
                value = value.to_dict()
            result[key] = value
        return result


class PlanNamespace:
    """
    Mimics the resolver, but instead of performing real work, it prints the steps that would
    be taken by the resolver when translating or calling algorithms
    """

    def __init__(self, resolver):
        self._resolver = resolver
        self.algos = Namespace()

    def translate(self, value, dst_type, **props):
        """
        Return translator to translate from type of value to dst_type
        """
        src_type = self._resolver.typeclass_of(value)
        translator = MultiStepTranslator.find_translation(
            self._resolver, src_type, dst_type
        )
        return translator

    def call_algorithm(self, algo_name: str, *args, **kwargs):
        valid_algos = self._resolver.find_algorithm_solutions(
            algo_name, *args, **kwargs
        )
        if not valid_algos:
            abstract_algo = self._resolver.abstract_algorithms[algo_name]
            sig = abstract_algo.__signature__
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            print(
                f'No concrete algorithm for "{algo_name}" can be satisfied for the given inputs'
            )
            for key, val in bound_args.arguments.items():
                print(f"{key} : {val.__class__.__name__}")
            print("-" * len(abstract_algo.__name__))
            print(abstract_algo.__name__)
            Dispatcher(self._resolver, algo_name).signatures
        else:
            # choose the solutions requiring the fewest translations
            plan = valid_algos[0]
            return plan

    @property
    def abstract_algorithms(self):
        return self._resolver.abstract_algorithms


class Resolver:
    """Manages a collection of plugins (types, translators, and algorithms).

    Provides utilities to resolve the types of objects, select translators,
    and dispatch to concrete algorithms based on type matching.
    """

    def __init__(self):
        self.abstract_types: Set[AbstractType] = set()
        self.concrete_types: Set[ConcreteType] = set()
        self.translators: Dict[Tuple[ConcreteType, ConcreteType], Translator] = {}

        # map abstract name to instance of abstract algorithm
        self.abstract_algorithms: Dict[str, AbstractAlgorithm] = {}
        self.abstract_algorithm_versions: Dict[str, Dict[int, AbstractAlgorithm]] = {}

        # map abstract name to set of concrete instances
        self.concrete_algorithms: DefaultDict[
            str, Set[ConcreteAlgorithm]
        ] = defaultdict(set)

        # map python classes to concrete types
        self.class_to_concrete: Dict[type, ConcreteType] = {}

        # translation graph matrices
        # Single-sourch shortest path matrix and predecessor matrix from scipy.sparse.csgraph.dijkstra
        self._translation_matrices: Dict[AbstractType, TranslationMatrix] = {}

        self.algos = Namespace()
        self.wrappers = Namespace()
        self.types = Namespace()

        self.plugins = Namespace()

        self.plan = PlanNamespace(self)

    def explore(self, embedded=None):
        from ..explorer import service

        if embedded is None:
            import asyncio

            loop = asyncio.get_event_loop()
            embedded = loop.is_running()

        return service.main(self, embedded)

    def register(self, plugins_by_name):
        """Register plugins for use with this resolver.

        Plugins will be processed in category order (see function signature)
        to ensure that abstract types are registered before concrete types,
        concrete types before translators, and so on.

        This function may be called multiple times to add additional plugins
        at any time.  Plugins cannot be removed. A plugin name may only be registered once.
        """
        # Build data structures for registration
        plugin_categories = (
            "abstract_types",
            "concrete_types",
            "wrappers",
            "translators",
            "abstract_algorithms",
            "concrete_algorithms",
        )
        items_by_plugin = {"all": defaultdict(set)}
        for plugin_name, plugin in plugins_by_name.items():
            items_by_plugin[plugin_name] = {}
            for cat in plugin_categories:
                items = set(plugin.get(cat, ()))
                if cat == "concrete_algorithms":
                    # Make a copy of concrete algorithms to avoid cross-mutation if multiple resolvers are used
                    items = {copy.copy(x) for x in items}
                items_by_plugin[plugin_name][cat] = items
                items_by_plugin["all"][cat] |= items

        self._register_plugin_attributes_in_tree(self, **items_by_plugin["all"])

        for plugin_name, plugin in plugins_by_name.items():
            if not plugin_name.isidentifier():
                raise ValueError(f"{repr(plugin_name)} is not a valid plugin name.")
            if hasattr(self.plugins, plugin_name):
                raise ValueError(f"{plugin_name} already registered.")
            # Initialize the plugin namespace
            self.plugins._register(plugin_name, Namespace())
            plugin_namespace = getattr(self.plugins, plugin_name)
            plugin_namespace._register("abstract_types", set())
            plugin_namespace._register("concrete_types", set())
            plugin_namespace._register("translators", {})
            plugin_namespace._register("abstract_algorithms", {})
            plugin_namespace._register("abstract_algorithm_versions", {})
            plugin_namespace._register("concrete_algorithms", defaultdict(set))
            plugin_namespace._register("algos", Namespace())
            plugin_namespace._register("wrappers", Namespace())
            plugin_namespace._register("types", Namespace())

            self._register_plugin_attributes_in_tree(
                plugin_namespace,
                **items_by_plugin[plugin_name],
                plugin_name=plugin_name,
            )

        return

    def _register_plugin_attributes_in_tree(
        self,
        tree: Union["Resolver", Namespace],
        abstract_types: Set[AbstractType] = set(),
        concrete_types: Set[ConcreteType] = set(),
        wrappers: Set[Wrapper] = set(),
        translators: Set[Translator] = set(),
        abstract_algorithms: Set[AbstractAlgorithm] = set(),
        concrete_algorithms: Set[ConcreteAlgorithm] = set(),
        plugin_name: Optional[str] = None,
    ):
        tree_is_resolver = self is tree
        tree_is_plugin = plugin_name is not None
        if not (tree_is_resolver or tree_is_plugin):
            raise ValueError("{tree} not known to be the resolver or a plugin.")

        for at in abstract_types:
            if tree_is_resolver and at in self.abstract_types:
                raise ValueError(f"abstract type {at.__qualname__} already exists")
            tree.abstract_types.add(at)

        if tree_is_resolver:
            # Validate unambiguous_subcomponents are registered and have sufficient properties
            # (must be done after all abstract types have been added above)
            for at in abstract_types:
                for usub in at.unambiguous_subcomponents:
                    if usub not in tree.abstract_types:
                        raise KeyError(
                            f"unambiguous subcomponent {usub.__qualname__} has not been registered"
                        )
                    missing_props = set(usub.properties) - set(at.properties)
                    if missing_props:
                        raise ValueError(
                            f"unambiguous subcomponent {usub.__qualname__} has additional properties beyond {at.__qualname__}"
                        )

        # Let concrete type associated with each wrapper be handled by concrete_types list
        concrete_types = set(
            concrete_types
        )  # copy; don't mutate original since we extend this set
        for wr in wrappers:
            # Wrappers without .Type had `register=False` and should not be registered
            if not hasattr(wr, "Type"):
                continue
            # Otherwise, register both the concrete type and the wrapper
            concrete_types.add(wr.Type)
            # Make wrappers available via resolver.wrappers.<abstract name>.<wrapper name>
            path = f"{wr.Type.abstract.__name__}.{wr.__name__}"
            tree.wrappers._register(path, wr)

        if tree_is_resolver and (len(concrete_types) > 0 or len(translators) > 0):
            # Wipe out existing translation matrices (if any)
            self._translation_matrices.clear()

        for ct in concrete_types:
            name = ct.__qualname__
            # ct.abstract cannot be None due to ConcreteType.__init_subclass__
            if tree_is_resolver:
                if ct.abstract not in self.abstract_types:
                    abstract_name = ct.abstract.__qualname__
                    raise ValueError(
                        f"concrete type {name} has unregistered abstract type {abstract_name}"
                    )
                if ct.value_type in self.class_to_concrete:
                    raise ValueError(
                        f"Python class '{ct.value_type}' already has a registered concrete type: {self.class_to_concrete[ct.value_type]}"
                    )
                if ct.value_type is not None:
                    self.class_to_concrete[ct.value_type] = ct

            tree.concrete_types.add(ct)

            # Make types available via resolver.types.<abstract name>.<concrete name>
            path = f"{ct.abstract.__name__}.{ct.__name__}"
            tree.types._register(path, ct)

        for tr in translators:
            signature = inspect.signature(tr.func)
            src_type = next(iter(signature.parameters.values())).annotation
            src_type = self.class_to_concrete.get(src_type, src_type)
            dst_type = signature.return_annotation
            dst_type = self.class_to_concrete.get(dst_type, dst_type)
            # Verify types are registered
            if src_type not in self.concrete_types:
                raise ValueError(
                    f"translator source type {src_type.__qualname__} has not been registered"
                )
            if dst_type not in self.concrete_types:
                raise ValueError(
                    f"translator destination type {dst_type.__qualname__} has not been registered"
                )
            # Verify translation is allowed
            if src_type.abstract != dst_type.abstract:
                # Check if dst is unambiguous subcomponent of src
                if dst_type.abstract not in src_type.abstract.unambiguous_subcomponents:
                    raise ValueError(
                        f"Translator {tr.func.__name__} must convert between concrete types of same abstract type ({src_type.abstract} != {dst_type.abstract})"
                    )
            tree.translators[(src_type, dst_type)] = tr

        for aa in abstract_algorithms:
            aa = self._normalize_abstract_algorithm_signature(aa)
            if aa.name not in tree.abstract_algorithm_versions:
                tree.abstract_algorithm_versions[aa.name] = {aa.version: aa}
                tree.abstract_algorithms[aa.name] = aa
                if tree_is_resolver:
                    self.algos._register(aa.name, Dispatcher(self, aa.name))
                    self.plan.algos._register(aa.name, Dispatcher(self.plan, aa.name))
            else:
                if (
                    tree_is_resolver
                    and aa.version in tree.abstract_algorithm_versions[aa.name]
                ):
                    raise ValueError(
                        f"abstract algorithm {aa.name} with version {aa.version} already exists"
                    )
                tree.abstract_algorithm_versions[aa.name][aa.version] = aa
                if aa.version > tree.abstract_algorithms[aa.name].version:
                    tree.abstract_algorithms[aa.name] = aa

        latest_concrete_versions = defaultdict(int)
        for ca in concrete_algorithms:
            if tree_is_resolver:
                abstract = self.abstract_algorithms.get(ca.abstract_name)
                if abstract is None:
                    raise ValueError(
                        f"concrete algorithm {ca.func.__module__}.{ca.func.__qualname__} "
                        f"implements unregistered abstract algorithm {ca.abstract_name}"
                    )
                if ca.version not in self.abstract_algorithm_versions[ca.abstract_name]:
                    action = config["core.algorithm.unknown_concrete_version"]
                    abstract_versions = ", ".join(
                        map(
                            str,
                            sorted(self.abstract_algorithm_versions[ca.abstract_name]),
                        )
                    )
                    message = (
                        f"concrete algorithm {ca.func.__module__}.{ca.func.__qualname__} implements "
                        f"an unknown version of abstract algorithm {ca.abstract_name}.\n\n"
                        f"The concrete version: {ca.version}.\n"
                        f"Abstract versions: {abstract_versions}"
                    )
                    if action is None or action == "ignore":
                        pass
                    elif action == "warn":
                        warnings.warn(message, AlgorithmWarning, stacklevel=2)
                    elif action == "raise":
                        raise ValueError(message)
                    else:
                        raise ValueError(
                            "Unknown configuration for 'core.algorithm.unknown_concrete_version'.\n"
                            f"Expected 'ignore', 'warn', or 'raise'.  Got: {action!r}.  Raising.\n\n"
                            + message
                        )
                latest_concrete_versions[ca.abstract_name] = max(
                    ca.version, latest_concrete_versions[ca.abstract_name]
                )
                if ca.version == abstract.version:
                    self._normalize_concrete_algorithm_signature(abstract, ca)
                else:
                    continue
            elif tree_is_plugin:
                abstract = self.abstract_algorithms.get(ca.abstract_name)
                if abstract is None or ca.version != abstract.version:
                    continue
                try:
                    # Register the exact algorithm call for resolver.plugin_name.algos.path.to.algo()
                    exact_dispatcher = ExactDispatcher(self, plugin_name, ca)
                    tree.algos._register(ca.abstract_name, exact_dispatcher)

                except NamespaceError:
                    raise ValueError(
                        f"Multiple concrete algorithms for abstract algorithm {ca.abstract_name} within plugin {plugin_name}."
                    )
                # Locate the abstract dispatcher
                dispatcher = self.algos
                for name in ca.abstract_name.split("."):
                    dispatcher = getattr(dispatcher, name)
                # Register the exact algorithm call for resolver.algos.path.to.algo.plugin_name()
                setattr(dispatcher, plugin_name, ExactDispatcher(self, plugin_name, ca))
            tree.concrete_algorithms[ca.abstract_name].add(ca)

        action = config["core.algorithms.outdated_concrete_version"]
        if action is not None and action != "ignore":
            for name, version in latest_concrete_versions.items():
                if version < self.abstract_algorithms[name].version:
                    message = (
                        f"concrete algorithm {ca.func.__module__}.{ca.func.__qualname__} implements "
                        f"an outdated version of abstract algorithm {ca.abstract_name}.\n\n"
                        f"The latest concrete version is {ca.version}.\n"
                        f"The latest abstract version is {abstract.version}."
                    )
                    if action == "warn":
                        warnings.warn(message, AlgorithmWarning, stacklevel=2)
                    elif action == "raise":
                        raise ValueError(message)
                    else:
                        raise ValueError(
                            "Unknown configuration for 'core.algorithm.outdated_concrete_version'.\n"
                            f"Expected 'ignore', 'warn', or 'raise'.  Got: {action!r}.  Raising.\n\n"
                            + message
                        )

    def _check_abstract_type(self, abst_algo, obj, msg):
        if obj is Any or obj is NodeID:
            return obj, False

        origin = getattr(obj, "__origin__", None)
        if origin == abc.Callable:
            return obj, False
        elif origin == Union:
            return mgtyping.Union[obj.__args__], True
        elif origin == list:
            if isinstance(obj.__args__[0], TypeVar):
                raise TypeError(
                    f"{abst_algo.func.__qualname__} {msg} must pass exactly one parameter to List."
                )
            return mgtyping.List[obj.__args__[0]], True

        if type(obj) is type:
            if issubclass(obj, AbstractType):
                return obj(), True
            # Non-abstract type class is assumed to be Python type
            return obj, False
        if isinstance(obj, mgtyping.Combo):
            if obj.kind not in {"python", "abstract", "node_id", "uniform_iterable"}:
                raise TypeError(
                    f"{abst_algo.func.__qualname__} {msg} may not have Concrete types in Union"
                )
            return obj, False
        elif isinstance(obj, mgtyping.UniformIterable):
            return obj, False
        if isinstance(obj, AbstractType):
            return obj, False

        # All valid cases listed are listed above; if we got here, raise an error
        wrong_type_str = f"an instance of type {type(obj)}"
        # Improve messaging for typing module objects
        if origin is not None and getattr(obj, "_name", None) is not None:
            wrong_type_str = f"typing.{obj._name}"
        raise TypeError(
            f"{abst_algo.func.__qualname__} {msg} may not be {wrong_type_str}"
        )

    def _normalize_abstract_algorithm_signature(self, abst_algo: AbstractAlgorithm):
        """
        Convert all AbstractType to a no-arg instance
        Leave all Python types alone
        Guard against instances of anything other than AbstractType
        """
        abs_sig = abst_algo.__signature__
        params = abs_sig.parameters
        ret = abs_sig.return_annotation
        params_modified = []
        any_changed = False
        for pname, p in params.items():
            pmod, changed = self._check_abstract_type(
                abst_algo, p.annotation, f'argument "{pname}"'
            )
            if changed:
                p = p.replace(annotation=pmod)
                any_changed = True
            params_modified.append(p)
        # Normalize return type, which might be a tuple
        if getattr(ret, "__origin__", None) == tuple:
            ret_modified = []
            for ret_sub in ret.__args__:
                ret_sub, changed = self._check_abstract_type(
                    abst_algo, ret_sub, "return type"
                )
                any_changed |= changed
                ret_modified.append(ret_sub)
            ret.__args__ = tuple(ret_modified)
        else:
            ret, changed = self._check_abstract_type(abst_algo, ret, "return type")
            any_changed |= changed

        if any_changed:
            abs_sig = abs_sig.replace(parameters=params_modified, return_annotation=ret)
            abst_algo.__signature__ = abs_sig

        return abst_algo

    def _normalize_concrete_type(self, conc_type, abst_type: AbstractType):
        changed = False
        origin = getattr(conc_type, "__origin__", None)

        if isinstance(abst_type, mgtyping.Combo) and not isinstance(
            conc_type, mgtyping.Combo
        ):
            if origin == Union:
                conc_type = mgtyping.Combo(conc_type.__args__, strict=abst_type.strict)
            else:
                conc_type = mgtyping.Combo(
                    [conc_type], optional=abst_type.optional, strict=abst_type.strict
                )
            changed = True
        elif isinstance(abst_type, mgtyping.UniformIterable) and not isinstance(
            conc_type, mgtyping.UniformIterable
        ):
            if origin == List:
                conc_type = mgtyping.List[conc_type.__args__[0]]
            else:
                conc_type = mgtyping.Combo(conc_type)
            changed = True
        elif isinstance(abst_type, AbstractType) and not isinstance(
            conc_type, ConcreteType
        ):
            if type(conc_type) is type and issubclass(conc_type, ConcreteType):
                return conc_type(), True
            # handle Python classes used as concrete types
            elif conc_type in self.class_to_concrete:
                conc_type = self.class_to_concrete[conc_type]()
                changed = True
            else:
                raise TypeError(
                    f"'{conc_type}' is not a concrete type of '{abst_type}'"
                )
        return conc_type, changed

    def _normalize_concrete_algorithm_signature(
        self, abstract: AbstractAlgorithm, concrete: ConcreteAlgorithm
    ):
        """
        Convert all ConcreteType to a no-arg instance
        Leave all Python types alone
        Guard against instances of anything other than ConcreteType
        Guard against mismatched signatures vs the abstract signature, while allowing
            for concrete signature to contain additional parameters beyond those defined
            in the abstract signature
        """
        abst_sig = abstract.__signature__
        conc_sig = concrete.__signature__

        # Check parameters
        abst_keys = set(abst_sig.parameters)
        abst_params = list(abst_sig.parameters.values())
        conc_params = list(conc_sig.parameters.values())

        # Check for missing parameters in concrete signature
        missing_params = set(abst_sig.parameters) - set(conc_sig.parameters)
        if missing_params:
            raise TypeError(
                f"Missing parameters: {missing_params} from {abstract.name} in implementation {concrete.func.__qualname__}"
            )

        # Walk through the parameters, which will not line up because the concrete may have extra parameters
        # Update concrete signature with defaults defined in the abstract signature
        # Verify that extra parameters contain a default value
        params_modified = []
        any_changed = False
        iabst = -1
        for iconc, conc_param in enumerate(conc_params):
            if conc_param.name == "resolver" and concrete._include_resolver:
                # Handle "include_resolver" logic; algo should not declare default,
                # but we add a default to make things easier for exact dispatching
                if conc_param.default is not inspect._empty:
                    raise TypeError('"resolver" should not have a default')
                conc_param = conc_param.replace(default=None)
                any_changed = True
            elif conc_param.name not in abst_keys:
                # Extra concrete params must declare a default value
                if conc_param.default is inspect._empty:
                    raise TypeError(
                        f'[{concrete.func.__qualname__}] argument "{conc_param.name}" is not found in abstract signature and must declare a default value'
                    )
            else:
                iabst += 1
                abst_param = abst_params[iabst]

                # Concrete parameters should never define a default value -- they inherit the default from the abstract signature
                if conc_param.default is not inspect._empty:
                    raise TypeError(
                        f'[{concrete.func.__qualname__}] argument "{conc_param.name}" declares a default value; default values can only be defined in the abstract signature'
                    )
                # If abstract defines a default, update concrete with the same default
                if abst_param.default is not inspect._empty:
                    conc_param = conc_param.replace(default=abst_param.default)
                    any_changed = True

                abst_type = abst_param.annotation
                conc_type, changed = self._normalize_concrete_type(
                    conc_type=conc_param.annotation, abst_type=abst_type
                )
                if changed:
                    conc_param = conc_param.replace(annotation=conc_type)
                    any_changed = True
                if abst_param.name != conc_param.name:
                    raise TypeError(
                        f'[{concrete.func.__qualname__}] argument "{conc_param.name}" does not match name of parameter in abstract function signature'
                    )

                if isinstance(conc_type, mgtyping.UniformIterable):
                    is_concrete = isinstance(conc_type.element_type, ConcreteType)
                    valid_concrete = is_concrete and issubclass(
                        conc_type.element_type.abstract,
                        abst_type.element_type.__class__,
                    )
                    valid_non_concrete = (
                        not is_concrete
                        and conc_type.element_type == abst_type.element_type
                    )
                    if not valid_concrete and not valid_non_concrete:
                        raise TypeError(f"{conc_type} does not match {abst_type}")
                elif isinstance(conc_type, mgtyping.Combo):
                    if abst_type.optional != conc_type.optional:
                        raise TypeError(
                            f"{conc_type} does not match optional flag in {abst_type}"
                        )
                    unmatched_types = []
                    # Verify that each item in conc_type matches at least one item in abst_type
                    for ct in conc_type.types:
                        is_concrete = isinstance(ct, ConcreteType)
                        for at in abst_type.types:
                            if is_concrete and issubclass(ct.abstract, at.__class__):
                                break
                            elif not is_concrete and ct == at:
                                break
                        else:
                            unmatched_types.append(ct)
                    if unmatched_types:
                        raise TypeError(f"{unmatched_types} not found in {abst_type}")
                elif isinstance(conc_type, ConcreteType):
                    if not issubclass(conc_type.abstract, abst_type.__class__):
                        raise TypeError(
                            f'{concrete.func.__qualname__} argument "{conc_param.name}" does not have type compatible with abstract function signature'
                        )
                    if conc_type.abstract_instance is not None:
                        raise TypeError(
                            f'{concrete.func.__qualname__} argument "{conc_param.name}" specifies abstract properties'
                        )
                else:
                    # regular Python types need to match exactly
                    if abst_type != conc_type:
                        raise TypeError(
                            f'{concrete.func.__qualname__} argument "{conc_param.name}" does not match abstract function signature'
                        )
            params_modified.append(conc_param)

        abst_ret = abst_sig.return_annotation
        conc_ret, changed = self._normalize_concrete_type(
            conc_type=conc_sig.return_annotation, abst_type=abst_ret
        )
        any_changed |= changed
        # Normalize return type, which might be a tuple
        if hasattr(conc_ret, "__origin__") and conc_ret.__origin__ == tuple:
            if len(abst_ret.__args__) != len(conc_ret.__args__):
                raise TypeError(
                    f"{concrete.func.__qualname__} return type is not compatible with abstract function signature"
                )
            ret_modified = []
            for conc_ret_sub_type, abst_ret_sub_type in zip(
                conc_ret.__args__, abst_ret.__args__
            ):
                conc_ret_sub_type_normalized, changed = self._normalize_concrete_type(
                    conc_type=conc_ret_sub_type, abst_type=abst_ret_sub_type
                )
                any_changed |= changed
                self._check_concrete_algorithm_return_signature(
                    concrete, conc_ret_sub_type_normalized, abst_ret_sub_type
                )
                ret_modified.append(conc_ret_sub_type_normalized)
            conc_ret.__args__ = tuple(ret_modified)
        else:
            self._check_concrete_algorithm_return_signature(
                concrete, conc_ret, abst_ret
            )
        if any_changed:
            conc_sig = conc_sig.replace(
                parameters=params_modified, return_annotation=conc_ret
            )
            concrete.__signature__ = conc_sig

    def _check_concrete_algorithm_return_signature(self, concrete, conc_ret, abst_ret):
        if isinstance(conc_ret, ConcreteType):
            if not issubclass(conc_ret.abstract, abst_ret.__class__):
                raise TypeError(
                    f"{concrete.func.__qualname__} return type is not compatible with abstract function signature"
                )
        else:
            # regular Python types need to match exactly
            if abst_ret != conc_ret:
                raise TypeError(
                    f"{concrete.func.__qualname__} return type does not match abstract function signature"
                )

    def load_plugins_from_environment(self):
        """Scans environment for plugins and populates registry with them."""
        plugins_by_name = load_plugins()
        self.register(plugins_by_name)

    def typeclass_of(self, value):
        """Return the concrete typeclass corresponding to a value"""
        # Check for direct lookup
        concrete_type = self.class_to_concrete.get(type(value))
        if concrete_type is None:
            for ct in self.concrete_types:
                if ct.is_typeclass_of(value):
                    concrete_type = ct
                    break
            else:
                raise TypeError(
                    f"Class {value.__class__} does not have a registered type"
                )
        return concrete_type

    def type_of(self, value):
        """Return the fully specified type for this value.

        This may require potentially slow computation of properties.  Only use
        this for debugging.
        """
        return self.typeclass_of(value).get_type(value)

    def assert_equal(self, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0):
        # Ensure all properties are fully calculated
        type1 = self.type_of(obj1)
        type2 = self.type_of(obj2)
        if type(type1) is not type(type2):
            raise TypeError(
                f"Cannot assert_equal with different types: {type(type1)} != {type(type2)}"
            )
        type1.assert_equal(
            obj1,
            obj2,
            type1.get_typeinfo(obj1).known_abstract_props,
            type2.get_typeinfo(obj2).known_abstract_props,
            type1.get_typeinfo(obj1).known_concrete_props,
            type2.get_typeinfo(obj2).known_concrete_props,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    def translate(self, value, dst_type, **props):
        """Convert a value to a new concrete type using translators"""
        src_type = self.typeclass_of(value)
        translator = MultiStepTranslator.find_translation(self, src_type, dst_type)
        if translator.unsatisfiable:
            raise TypeError(f"Cannot convert {value} to {dst_type}")
        return translator(value, **props)

    def find_algorithm_solutions(
        self, algo_name: str, *args, **kwargs
    ) -> List[AlgorithmPlan]:
        if algo_name not in self.abstract_algorithms:
            raise ValueError(f'No abstract algorithm "{algo_name}" has been registered')

        # Find all possible solution paths
        solutions: List[AlgorithmPlan] = []
        for concrete_algo in self.concrete_algorithms.get(algo_name, {}):
            plan = AlgorithmPlan.build(self, concrete_algo, *args, **kwargs)
            if not plan.unsatisfiable:
                solutions.append(plan)

        # Sort by fewest number of translations required
        def total_num_translations(plan):
            return sum(len(t) for t in plan.required_translations.values())

        # TODO: improve this in the future. for now, use total number of translations
        #       as well as algorithm name to ensure repeatability of solutions
        solutions.sort(key=lambda x: (total_num_translations(x), x.algo.func.__name__))

        return solutions

    def find_algorithm_exact(
        self, algo_name: str, *args, **kwargs
    ) -> Optional[ConcreteAlgorithm]:
        valid_algos = self.find_algorithm_solutions(algo_name, *args, **kwargs)
        if valid_algos:
            best_algo = valid_algos[0]
            if not best_algo.required_translations:
                return best_algo

    def find_algorithm(
        self, algo_name: str, *args, **kwargs
    ) -> Optional[ConcreteAlgorithm]:
        valid_algos = self.find_algorithm_solutions(algo_name, *args, **kwargs)
        if valid_algos:
            best_algo = valid_algos[0]
            return best_algo

    def _check_valid_arg(self, arg_name, arg_value, param_type):
        if param_type is Any:
            return
        param_class = type(param_type)
        if param_class is type:
            if not isinstance(arg_value, param_type):
                return (
                    f"{arg_name} must be of type {param_type.__name__}, "
                    f"not {type(arg_value).__name__}"
                )
        if isinstance(param_type, AbstractType):
            try:
                this_typeclass = self.typeclass_of(arg_value)
            except TypeError:
                return f"{arg_name} must be of type {param_class.__name__}, not {type(arg_value)}"

            # Check if arg_value has the right abstract type
            if this_typeclass.abstract != param_class:
                # Allow for unambiguous subcomponent
                if param_class not in this_typeclass.abstract.unambiguous_subcomponents:
                    return (
                        f"{arg_name} must be of type {param_class.__name__}, "
                        f"not {this_typeclass.abstract.__name__}::{this_typeclass.__name__}"
                    )

            requested_properties = set(
                k for k, v in param_type.prop_val.items() if v is not None
            )
            properties_dict = this_typeclass.compute_abstract_properties(
                arg_value, requested_properties
            )
            this_abs_type = this_typeclass.abstract(**properties_dict)

            unsatisfied_requirements = []
            for abst_prop, required_value in param_type.prop_val.items():
                if required_value is None:  # unspecified
                    continue
                if type(required_value) is tuple:
                    if this_abs_type.prop_val[abst_prop] not in required_value:
                        unsatisfied_requirements.append(
                            f" -> `{abst_prop}` must be one of {required_value!r}"
                        )
                else:
                    if this_abs_type.prop_val[abst_prop] != required_value:
                        unsatisfied_requirements.append(
                            f" -> `{abst_prop}` must be {required_value!r}"
                        )
            if unsatisfied_requirements:
                return (
                    f'"{arg_name}" with properties\n{this_abs_type.prop_val}\n'
                    + f"does not meet requirements:\n"
                    + "\n".join(unsatisfied_requirements)
                )

    def _check_algorithm_signature(
        self, algo_name: str, *args, allow_extras=False, **kwargs
    ):
        if algo_name not in self.abstract_algorithms:
            raise ValueError(f'No abstract algorithm "{algo_name}" has been registered')

        # Validate types have required abstract properties
        abstract_algo = self.abstract_algorithms[algo_name]
        sig = abstract_algo.__signature__
        extra_args, extra_kwargs = [], {}
        if allow_extras:
            # Try to bind signature, removing extra parameters until satisfied
            while True:
                try:
                    bound_args = sig.bind(*args, **kwargs)
                    break
                except TypeError as e:
                    if e.args:
                        if e.args[0] == "too many positional arguments":
                            extra_args.insert(0, args[-1])
                            args = args[:-1]
                            continue
                        elif e.args[0][:34] == "got an unexpected keyword argument":
                            key = e.args[0][36:-1]
                            extra_kwargs[key] = kwargs.pop(key)
                            continue
                    raise
        else:
            bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        parameters = bound_args.signature.parameters
        for arg_name, arg_value in bound_args.arguments.items():
            param_type = parameters[arg_name].annotation
            if isinstance(param_type, mgtyping.Combo):
                if arg_value is None:
                    if param_type.optional:
                        continue
                    else:
                        raise TypeError(
                            f"{arg_name} is None, but the parameter is not Optional"
                        )

                # Find any satisfiable value in the Combo
                for pt in param_type.types:
                    err_msg = self._check_valid_arg(arg_name, arg_value, pt)
                    if not err_msg:
                        break
                else:
                    raise TypeError(
                        f"{arg_name} (type={type(arg_value)}) does not match any of {param_type}"
                    )
            else:
                err_msg = self._check_valid_arg(arg_name, arg_value, param_type)
                if err_msg:
                    raise TypeError(err_msg)

        if extra_args or extra_kwargs:
            return (
                bound_args.args + tuple(extra_args),
                {**bound_args.kwargs, **extra_kwargs},
            )
        else:
            return bound_args.args, bound_args.kwargs

    def call_algorithm(self, algo_name: str, *args, **kwargs):
        args, kwargs = self._check_algorithm_signature(algo_name, *args, **kwargs)

        if config.get("core.dispatch.allow_translation"):
            algo = self.find_algorithm(algo_name, *args, **kwargs)
        else:
            algo = self.find_algorithm_exact(algo_name, *args, **kwargs)

        if not algo:
            raise TypeError(
                f'No concrete algorithm for "{algo_name}" can be satisfied for the given inputs'
            )

        if config.get("core.logging.plans"):
            algo.display()
        return algo(*args, **kwargs)

    def call_exact_algorithm(self, concrete_algo: ConcreteAlgorithm, *args, **kwargs):
        args, kwargs = self._check_algorithm_signature(
            concrete_algo.abstract_name, *args, allow_extras=True, **kwargs
        )
        plan = AlgorithmPlan.build(self, concrete_algo, *args, **kwargs)
        if plan.unsatisfiable:
            err_msgs = "\n".join(plan.err_msgs)
            raise TypeError(
                f"Incorrect input types and no valid translation path to solution.\n{err_msgs}"
            )
        elif plan.required_translations:
            req_trans = ", ".join(plan.required_translations.keys())
            raise TypeError(
                f"Incorrect input types. Translations required for: {req_trans}"
            )
        else:
            return plan(*args, **kwargs)


class Dispatcher:
    """Impersonates abstract algorithm, but dispatches to a resolver to select
    the appropriate concrete algorithm."""

    def __init__(self, resolver: Resolver, algo_name: str):
        self._resolver = resolver
        self._algo_name = algo_name

        # make dispatcher look like the abstract algorithm
        abstract_algo = resolver.abstract_algorithms[algo_name].func
        self.__name__ = algo_name
        self.__doc__ = abstract_algo.__doc__
        self.__signature__ = inspect.signature(abstract_algo)
        self.__wrapped__ = abstract_algo

    def __call__(self, *args, **kwargs):
        return self._resolver.call_algorithm(self._algo_name, *args, **kwargs)

    @property
    def signatures(self):
        print("Signature:")
        print(f"\t{self.__signature__}")
        print("Implementations:")
        for ca in self._resolver.concrete_algorithms[self._algo_name]:
            # print(f"\t{ca.func.__annotations__}")
            print(f"\t{ca.__signature__}")


class ExactDispatcher:
    """Impersonates concrete algorithm, but dispatches to a resolver to verify
    the concrete algorithm inputs prior to calling the function."""

    def __init__(self, resolver: Resolver, plugin: str, algo: ConcreteAlgorithm):
        self._resolver = resolver
        self._plugin = plugin
        self._algo = algo

        # make dispatcher look like the concrete algorithm
        self.__name__ = algo.abstract_name
        self.__doc__ = algo.func.__doc__
        self.__signature__ = inspect.signature(algo.func)
        self.__wrapped__ = algo

    def __call__(self, *args, **kwargs):
        return self._resolver.call_exact_algorithm(self._algo, *args, **kwargs)
