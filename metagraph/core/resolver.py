"""A Resolver manages a collection of plugins, resolves types, and dispatches
to concrete algorithms.

"""
import copy
import inspect
import warnings
from collections import defaultdict, abc
import typing
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
    Compiler,
    CompileError,
)
from .planning import MultiStepTranslator, AlgorithmPlan, TranslationMatrix
from .entrypoints import load_plugins
from . import typing as mgtyping
from .. import config
from .typing import NodeID
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
        dst_type = self._resolver._normalize_translation_destination(dst_type, src_type)
        translator = MultiStepTranslator.find_translation(
            self._resolver, src_type, dst_type
        )
        return translator

    def run(self, algo_name: str, *args, **kwargs):
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
    """
    Manages a collection of plugins (types, translators, and algorithms).

    Provides utilities to resolve the types of objects, select translators,
    and dispatch to concrete algorithms based on type matching.

    Can be used as a context manager to set the default resolver (when using custom resolvers).
    For example:
       cust_resolver = Resolver()
       # .. register things with cust_resolver
       with cust_resolver:
           my_graph.run('centrality.pagerank')
    """

    def __init__(self):
        self.abstract_types: Set[AbstractType] = set()
        self.concrete_types: Set[ConcreteType] = set()
        self.translators: Dict[Tuple[ConcreteType, ConcreteType], Translator] = {}
        self.compilers: Dict[str, Compiler] = {}

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
        # Single-source shortest path matrix and predecessor matrix from scipy.sparse.csgraph.dijkstra
        self._translation_matrices: Dict[AbstractType, TranslationMatrix] = {}

        self.algos = Namespace()
        self.wrappers = Namespace()
        self.types = Namespace()

        self.plugins = Namespace()

        self.plan = PlanNamespace(self)

    def __enter__(self):
        self.set_as_default()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reset_default()

    def set_as_default(self):
        import metagraph as mg

        if not hasattr(self, "_resolver_stack"):
            self._resolver_stack = []
        # Save the current default
        self._resolver_stack.append(mg.resolver)
        # Set myself as the new default
        mg._set_default_resolver(self)

    def reset_default(self):
        import metagraph as mg

        # Reset the default resolver to the previous value
        prev = self._resolver_stack.pop()
        mg._set_default_resolver(prev)

    def explore(self, embedded=None):
        from ..explorer import service

        if embedded is None:
            import asyncio

            loop = asyncio.get_event_loop()
            embedded = loop.is_running()

        return service.main(self, embedded)

    def register(self, plugins_by_name):
        """Register plugins for use with a resolver."""
        _ResolverRegistrar.register(self, plugins_by_name)

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

    def _find_translatable_concrete_type_by_name(
        self, name: str, starting_type: AbstractType
    ):
        """
        Given a starting_type, attempts to find a ConcreteType matching name
        Resolution order:
        1. ConcreteTypes for starting_type
        2. Wrappers for starting_type
        3. ConcreteTypes for all unambiguous_subcomponents of starting_type
        4. Wrappers for all unambiguous_subcomponents of starting_type
        As soon as a match is found, it is returned. This method will not
        check for possible duplicate names.
        Raises AttributeError if name is not found
        """
        # Interpret "FooBar.Type" instead of "FooBarType"
        if "." in name and name.endswith(".Type"):
            name = f"{name[:-5]}Type"

        # Check direct concrete types
        cat = getattr(self.types, starting_type.__name__)
        for ct in dir(cat):
            if ct == name:
                return getattr(cat, ct)
        # Check direct wrappers
        wcat = getattr(self.wrappers, starting_type.__name__)
        for wr in dir(wcat):
            if wr == name:
                return getattr(wcat, wr).Type
        # Check secondary concrete types
        for subtype in starting_type.unambiguous_subcomponents:
            cat = getattr(self.types, subtype.__name__)
            for ct in dir(cat):
                if ct == name:
                    return getattr(cat, ct)
        # Check secondary wrappers
        for subtype in starting_type.unambiguous_subcomponents:
            wcat = getattr(self.wrappers, subtype.__name__)
            for wr in dir(wcat):
                if wr == name:
                    return getattr(wcat, wr).Type
        # Not found, raise
        raise AttributeError(
            f'No translatable type found for "{name}" within {starting_type}'
        )

    def _normalize_translation_destination(self, dst_type, src_type):
        # Normalize dst_type, which could be:
        #  - Wrapper, instance of Wrapper, or string of Wrapper class name
        #  - ConcreteType or string of ConcreteType class name
        #  - ConcreteType's value_type or instance of ConcreteType's value_type
        orig_dst_type = dst_type
        if not isinstance(dst_type, type):
            if isinstance(dst_type, str):
                dst_type = self._find_translatable_concrete_type_by_name(
                    dst_type, src_type.abstract
                )
            elif isinstance(dst_type, Wrapper):
                dst_type = dst_type.Type
            else:
                dst_type = type(dst_type)

        assert isinstance(dst_type, type)
        if issubclass(dst_type, Wrapper):
            dst_type = dst_type.Type
        elif not issubclass(dst_type, ConcreteType):
            dst_type = self.class_to_concrete.get(dst_type, dst_type)
            if not issubclass(dst_type, ConcreteType):
                raise TypeError(f"Unexpected dst_type: {orig_dst_type}")
        return dst_type

    def translate(self, value, dst_type: Union[str, ConcreteType, Wrapper], **props):
        """Convert a value to a new concrete type using translators"""
        src_type = self.typeclass_of(value)
        dst_type = self._normalize_translation_destination(dst_type, src_type)
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

    def run(self, algo_name: str, *args, **kwargs):
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

    def _check_algorithm_signature(
        self, algo_name: str, *args, allow_extras=False, **kwargs
    ):
        """
        Binds the variables from args and kwargs to those of the concrete algorithm.
        Checks that types match signature.
        """
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
                    raise  # pragma: no cover
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

                if param_type.kind == "List":
                    if not isinstance(arg_value, (list, tuple)):
                        raise TypeError(
                            f"{arg_name} must be a list, not {type(arg_name)}"
                        )
                    args_to_check = [
                        (f"{arg_name}[{iv}]", v) for iv, v in enumerate(arg_value)
                    ]
                else:
                    args_to_check = [(arg_name, arg_value)]
                for name, val in args_to_check:
                    # Find any satisfiable value in the Combo
                    for pt in param_type.types:
                        err_msg = self._check_valid_arg(name, val, pt)
                        if not err_msg:
                            break
                    else:
                        raise TypeError(
                            f"{name} (type={type(val)}) does not match any of {param_type}"
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

    def compile_algorithm(
        self, concrete_algo: ConcreteAlgorithm, literals: Dict[str, Any] = None
    ) -> Callable:
        compiler_name = concrete_algo._compiler
        if compiler_name is None:
            raise CompileError(
                f"Concrete algorithm '{concrete_algo.__name__}' is not compilable"
            )

        compiler = self.compilers.get(compiler_name, None)
        if compiler is None:
            raise CompileError(f"Required compiler '{compiler_name}' not found")

        func = compiler.compile_algorithm(concrete_algo, literals=literals)
        return func


class _ResolverRegistrar:
    """
    Static methods to register plugins for use with a resolver.
    """

    @classmethod
    def register(cls, resolver: Resolver, plugins_by_name) -> None:
        """Register plugins for use with a resolver.

        Plugins will be processed in category order (see register_plugin_attributes_in_tree)
        to ensure that abstract types are registered before concrete types,
        concrete types before translators, etc.

        This may be called multiple times to add additional plugins at any time. 
        Plugins cannot be removed. 
        A plugin name may only be registered once.
        """

        # Build data structures for registration
        plugin_categories = (
            "abstract_types",
            "concrete_types",
            "wrappers",
            "translators",
            "abstract_algorithms",
            "concrete_algorithms",
            "compilers",
        )

        items_by_plugin = {"all": defaultdict(set)}
        for plugin_name, plugin in plugins_by_name.items():
            items_by_plugin[plugin_name] = {}
            for cat in plugin_categories:
                items = set(plugin.get(cat, ()))
                if cat in ("concrete_algorithms", "translators"):
                    # Copy to avoid cross-mutation if registered with multiple resolvers
                    items = {x.copy_and_bind(resolver) for x in items}
                items_by_plugin[plugin_name][cat] = items
                items_by_plugin["all"][cat] |= items

        cls.register_plugin_attributes_in_tree(
            resolver, resolver, **items_by_plugin["all"]
        )

        for plugin_name, plugin in plugins_by_name.items():
            if not plugin_name.isidentifier():
                raise ValueError(f"{repr(plugin_name)} is not a valid plugin name.")
            if hasattr(resolver.plugins, plugin_name):
                raise ValueError(f"{plugin_name} already registered.")
            # Initialize the plugin namespace
            resolver.plugins._register(plugin_name, Namespace())
            plugin_namespace = getattr(resolver.plugins, plugin_name)
            plugin_namespace._register("abstract_types", set())
            plugin_namespace._register("concrete_types", set())
            plugin_namespace._register("translators", {})
            plugin_namespace._register("abstract_algorithms", {})
            plugin_namespace._register("abstract_algorithm_versions", {})
            plugin_namespace._register("concrete_algorithms", defaultdict(set))
            plugin_namespace._register("compilers", {})
            plugin_namespace._register("algos", Namespace())
            plugin_namespace._register("wrappers", Namespace())
            plugin_namespace._register("types", Namespace())

            cls.register_plugin_attributes_in_tree(
                plugin_namespace,
                resolver,
                **items_by_plugin[plugin_name],
                plugin_name=plugin_name,
            )

        return

    @classmethod
    def register_plugin_attributes_in_tree(
        cls,
        tree: Union[Resolver, Namespace],
        resolver: Resolver,
        abstract_types: Set[AbstractType] = set(),
        concrete_types: Set[ConcreteType] = set(),
        wrappers: Set[Wrapper] = set(),
        translators: Set[Translator] = set(),
        abstract_algorithms: Set[AbstractAlgorithm] = set(),
        concrete_algorithms: Set[ConcreteAlgorithm] = set(),
        compilers: Set[Compiler] = set(),
        plugin_name: Optional[str] = None,
    ) -> None:
        """
        This method is intended to register attributes for a tree, 
        which is either a resolver or a plugin in a resolver.
        """

        # tree_is_resolver is used to avoid raising exceptions when attributes (e.g. abstract algorithms,
        # concrete types, translators, etc.) are redundantly registered. Exceptions should only be
        # raised when they're redundantly registered on the resolver. We'll necessarily have to
        # redundantly register certain attributes when we're registering them on a plugin since
        # they'll already have been registered in the resolver.
        tree_is_resolver = resolver is tree
        tree_is_plugin = plugin_name is not None

        if not (tree_is_resolver ^ tree_is_plugin):
            # the tree must either be a resolver or plugin (but not both)
            # if it's a plugin, then this method assumes plugin_name is provided (i.e. is not None).
            raise ValueError("{tree} not known to be the resolver or a plugin.")

        for at in abstract_types:
            if tree_is_resolver and at in resolver.abstract_types:
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
                            f"unambiguous subcomponent {usub.__qualname__} has additional "
                            f"properties beyond {at.__qualname__}"
                        )

        # Let concrete type associated with each wrapper be handled by concrete_types list
        concrete_types = set(
            concrete_types
        )  # copy; don't mutate original since we extend this set

        cls.register_wrappers_in_tree(tree, concrete_types, wrappers)

        if tree_is_resolver and (len(concrete_types) > 0 or len(translators) > 0):
            # Wipe out existing translation matrices (if any)
            resolver._translation_matrices.clear()

        cls.register_concrete_types_in_tree(tree, resolver, concrete_types)
        cls.register_translators_in_tree(tree, resolver, translators)
        cls.register_abstract_algorithms_in_tree(tree, resolver, abstract_algorithms)

        if tree_is_resolver:
            for compiler in compilers:
                if compiler.name in tree.compilers:
                    existing_compiler = tree.compilers[compiler.name]
                    raise ValueError(
                        f"Cannot register compiler named '{compiler.name}' from {compiler.__class__}\n"
                        f" when {existing_compiler.__class__} has already been registered with the same name."
                    )
                tree.compilers[compiler.name] = compiler

        cls.register_concrete_algorithms_in_tree(
            tree, resolver, concrete_algorithms, plugin_name
        )

        return

    @classmethod
    def register_wrappers_in_tree(
        cls,
        tree: Union[Resolver, Namespace],
        concrete_types: Set[ConcreteType],
        wrappers: Set[Wrapper],
    ) -> None:
        """
        Helper for register_plugin_attributes_in_tree to solely register wrappers.
        This method modifies concrete_types by adding elements to it.
        """
        for wr in wrappers:
            # Wrappers without .Type had `register=False` and should not be registered
            if not hasattr(wr, "Type"):
                continue
            # Otherwise, register both the concrete type and the wrapper
            concrete_types.add(wr.Type)
            # Make wrappers available via resolver.wrappers.<abstract name>.<wrapper name>
            path = f"{wr.Type.abstract.__name__}.{wr.__name__}"
            tree.wrappers._register(path, wr)

        return

    @classmethod
    def register_concrete_types_in_tree(
        cls,
        tree: Union[Resolver, Namespace],
        resolver: Resolver,
        concrete_types: Set[ConcreteType],
    ) -> None:
        """
        Helper for register_plugin_attributes_in_tree to solely register concrete types.
        """
        tree_is_resolver = resolver is tree

        for ct in concrete_types:
            name = ct.__qualname__
            # ct.abstract cannot be None due to ConcreteType.__init_subclass__
            if tree_is_resolver:
                if ct.abstract not in resolver.abstract_types:
                    abstract_name = ct.abstract.__qualname__
                    raise ValueError(
                        f"concrete type {name} has unregistered abstract type {abstract_name}"
                    )
                if ct.value_type in resolver.class_to_concrete:
                    raise ValueError(
                        f"Python class '{ct.value_type}' already has a registered "
                        f"concrete type: {resolver.class_to_concrete[ct.value_type]}"
                    )
                if ct.value_type is not None:
                    resolver.class_to_concrete[ct.value_type] = ct

            tree.concrete_types.add(ct)

            # Make types available via resolver.types.<abstract name>.<concrete name>
            path = f"{ct.abstract.__name__}.{ct.__name__}"
            tree.types._register(path, ct)

        return

    @classmethod
    def register_translators_in_tree(
        cls,
        tree: Union[Resolver, Namespace],
        resolver: Resolver,
        translators: Set[Translator],
    ) -> None:
        """
        Helper for register_plugin_attributes_in_tree to solely register translators.
        """
        for tr in translators:
            signature = inspect.signature(tr.func)
            src_type = next(iter(signature.parameters.values())).annotation
            src_type = resolver.class_to_concrete.get(src_type, src_type)
            dst_type = signature.return_annotation
            dst_type = resolver.class_to_concrete.get(dst_type, dst_type)
            # Verify types are registered
            if src_type not in resolver.concrete_types:
                raise ValueError(
                    f"translator source type {src_type.__qualname__} has not been registered"
                )
            if dst_type not in resolver.concrete_types:
                raise ValueError(
                    f"translator destination type {dst_type.__qualname__} has not been registered"
                )
            # Verify translation is allowed
            if src_type.abstract != dst_type.abstract:
                # Check if dst is unambiguous subcomponent of src
                if dst_type.abstract not in src_type.abstract.unambiguous_subcomponents:
                    raise ValueError(
                        f"Translator {tr.func.__name__} must convert between concrete types "
                        f"of same abstract type ({src_type.abstract} != {dst_type.abstract})"
                    )
            tree.translators[(src_type, dst_type)] = tr

        return

    @classmethod
    def register_abstract_algorithms_in_tree(
        cls,
        tree: Union[Resolver, Namespace],
        resolver: Resolver,
        abstract_algorithms: Set[AbstractAlgorithm],
    ) -> None:
        """
        Helper for register_plugin_attributes_in_tree to solely register abstract algorithms.
        """
        tree_is_resolver = resolver is tree

        for aa in abstract_algorithms:
            cls.normalize_abstract_algorithm_signature(aa)
            if aa.name not in tree.abstract_algorithm_versions:
                tree.abstract_algorithm_versions[aa.name] = {aa.version: aa}
                tree.abstract_algorithms[aa.name] = aa
                if tree_is_resolver:
                    resolver.algos._register(aa.name, Dispatcher(resolver, aa.name))
                    resolver.plan.algos._register(
                        aa.name, Dispatcher(resolver.plan, aa.name)
                    )
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
        return

    @classmethod
    def register_concrete_algorithms_in_tree(
        cls,
        tree: Union[Resolver, Namespace],
        resolver: Resolver,
        concrete_algorithms: Set[ConcreteAlgorithm],
        plugin_name: Optional[str],
    ) -> None:
        """
        Helper for register_plugin_attributes_in_tree to solely register concrete algorithms.
        """
        tree_is_resolver = resolver is tree
        tree_is_plugin = plugin_name is not None

        latest_concrete_versions = defaultdict(int)

        for ca in concrete_algorithms:
            if tree_is_resolver:
                abstract = resolver.abstract_algorithms.get(ca.abstract_name)
                if abstract is None:
                    raise ValueError(
                        f"concrete algorithm {ca.func.__module__}.{ca.func.__qualname__} "
                        f"implements unregistered abstract algorithm {ca.abstract_name}"
                    )
                if (
                    ca.version
                    not in resolver.abstract_algorithm_versions[ca.abstract_name]
                ):
                    action = config["core.algorithm.unknown_concrete_version"]
                    abstract_versions = ", ".join(
                        map(
                            str,
                            sorted(
                                resolver.abstract_algorithm_versions[ca.abstract_name]
                            ),
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
                    elif action == "warn":  # pragma: no cover
                        warnings.warn(message, AlgorithmWarning, stacklevel=2)
                    elif action == "raise":
                        raise ValueError(message)
                    else:  # pragma: no cover
                        raise ValueError(
                            "Unknown configuration for 'core.algorithm.unknown_concrete_version'.\n"
                            f"Expected 'ignore', 'warn', or 'raise'.  Got: {action!r}.  Raising.\n\n"
                            + message
                        )
                latest_concrete_versions[ca.abstract_name] = max(
                    ca.version, latest_concrete_versions[ca.abstract_name]
                )
                if ca.version != abstract.version:
                    continue
                else:
                    cls.normalize_concrete_algorithm_signature(resolver, abstract, ca)
            elif tree_is_plugin:
                abstract = resolver.abstract_algorithms.get(ca.abstract_name)
                if abstract is None or ca.version != abstract.version:
                    continue
                try:
                    # Register the exact algorithm call for resolver.plugin_name.algos.path.to.algo()
                    exact_dispatcher = ExactDispatcher(resolver, plugin_name, ca)
                    tree.algos._register(ca.abstract_name, exact_dispatcher)

                except NamespaceError:
                    raise ValueError(
                        f"Multiple concrete algorithms for abstract algorithm {ca.abstract_name} "
                        f"within plugin {plugin_name}."
                    )
                # Locate the abstract dispatcher
                dispatcher = resolver.algos
                for name in ca.abstract_name.split("."):
                    dispatcher = getattr(dispatcher, name)
                # Register the exact algorithm call for resolver.algos.path.to.algo.plugin_name()
                setattr(
                    dispatcher, plugin_name, ExactDispatcher(resolver, plugin_name, ca)
                )
            tree.concrete_algorithms[ca.abstract_name].add(ca)

        # Check for concrete algorithms implementing outdated abstract algorithms
        action = config["core.algorithms.outdated_concrete_version"]
        if action is not None and action != "ignore":
            for name, version in latest_concrete_versions.items():
                if version < resolver.abstract_algorithms[name].version:
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
                    else:  # pragma: no cover
                        raise ValueError(
                            "Unknown configuration for 'core.algorithm.outdated_concrete_version'.\n"
                            f"Expected 'ignore', 'warn', or 'raise'.  Got: {action!r}.  Raising.\n\n"
                            + message
                        )
        return

    @classmethod
    def normalize_abstract_algorithm_signature(
        cls, abst_algo: AbstractAlgorithm
    ) -> None:
        """
        This method "normalizes" the parameter types in the abstract algorithm signature.
        "Normalizes" means:
            * A parameter and its type is typically declared 
              like "abstract_algo_name(graph: Graph(is_directed=False))"
            * This method "executes" or instantiates the type declaration "Graph(is_directed=False)" 
              to create an actual type.
        This method also does the same with the concrete and abstract return types.
        This method handles "combination" types as well.

        This method modifies abst_algo.
        
        Convert all AbstractType to a no-arg instance
        Leave all Python types alone
        Guard against instances of anything other than AbstractType
        """
        abs_sig = abst_algo.__signature__
        params = abs_sig.parameters
        sig_mod = _SignatureModifier(abst_algo)
        for pname, p in params.items():
            cls.normalize_and_check_abstract_type(
                p.annotation, abst_algo, sig_mod, name=pname
            )

        # Normalize return type, which might be a tuple
        ret = abs_sig.return_annotation
        if typing.get_origin(ret) is tuple:
            for ret_sub_index, ret_sub in enumerate(typing.get_args(ret)):
                cls.normalize_and_check_abstract_type(
                    ret_sub, abst_algo, sig_mod, index=ret_sub_index
                )
        else:
            cls.normalize_and_check_abstract_type(ret, abst_algo, sig_mod)

        return

    @classmethod
    def normalize_and_check_abstract_type(
        cls,
        obj,
        abst_algo: AbstractAlgorithm,
        sig_mod: Union["_SignatureModifier", mgtyping.Combo],
        *,
        name=None,
        index=None,
    ):
        """
        This is a helper for normalize_abstract_algorithm_signature.
        If we have an abstract algorithm declaration like 
        "abstract_algo_name(graph: Graph(is_directed=False))", this method does the 
        actual "instantiation" of a type declaration like "Graph(is_directed=False)" 
        into an actual type.
        """
        msg = (
            abst_algo.func.__qualname__
            + " "
            + (f'argument "{name}"' if name is not None else "return type")
        )

        if obj is Any or obj is NodeID:
            return

        # Convert normal typing objects into Combos
        origin = typing.get_origin(obj)
        if origin == abc.Callable:
            return
        elif origin == Union:
            obj = mgtyping.Union[typing.get_args(obj)]
            sig_mod.update_annotation(obj, name=name, index=index)
        elif origin == list:
            obj = mgtyping.List[typing.get_args(obj)]
            sig_mod.update_annotation(obj, name=name, index=index)

        if type(obj) is type:
            if issubclass(obj, AbstractType):
                sig_mod.update_annotation(obj(), name=name, index=index)
                return
            elif issubclass(obj, ConcreteType):
                raise TypeError(f"{msg} may not have Concrete types in signature")
            else:
                # Non-abstract and non-concrete type class is assumed to be Python type
                pass
            return
        elif isinstance(obj, mgtyping.Combo):
            if isinstance(sig_mod, mgtyping.Combo):
                # Optional[List] or Optional[Union] is allowed
                if sig_mod.kind is None and sig_mod.optional and not obj.optional:
                    sig_mod.kind = obj.kind
                    sig_mod.types = obj.types
                    obj = sig_mod
                else:
                    # Everything else is not allowed
                    raise TypeError(
                        "Nesting a Combo type inside a Combo type is not allowed"
                    )
            # Normalize and check nested types
            for combo_idx, typ_ in enumerate(obj.types):
                # obj (a Combo) masquerades as sig_mod during this call
                cls.normalize_and_check_abstract_type(
                    typ_, abst_algo, sig_mod=obj, name=name, index=combo_idx
                )
            # This must be done after all nested types have been upgraded
            obj.compute_common_subtype()
            return
        elif isinstance(obj, AbstractType):
            return

        # All valid cases listed are listed above; if we got here, raise an error
        wrong_type_str = f"an instance of type {type(obj)}"
        # Improve messaging for typing module objects
        if origin is not None and getattr(obj, "_name", None) is not None:
            wrong_type_str = f"typing.{obj._name}"
        raise TypeError(f"{msg} may not be {wrong_type_str}")

    @classmethod
    def normalize_concrete_algorithm_signature(
        cls,
        resolver: Resolver,
        abstract: AbstractAlgorithm,
        concrete: ConcreteAlgorithm,
    ) -> None:
        """
        This method checks that the concrete and abstract signatures match, e.g. 
            * no missing params
            * concrete types of the concrete parameters match the abstract types of 
              the abstract parameter types
            * checks that no default parameters are provided in the concrete 
              algorithm (besides backend-specific parameters)
            * recursively checks for matches in "combination" types, e.g. list types, 
              tuple types, optional types, etc.
                * NB: this only recursively goes one step down, e.g. 
                  Tuple[Tuple[Tuple[int]]] is intentionally not supported.
        This method also does the same with the concrete and abstract return types.
        This method is used when registering concrete algorithms into a resolver.
        
        Convert all ConcreteType to a no-arg instance
        Leave all Python types alone
        Guard against instances of anything other than ConcreteType
        Guard against mismatched signatures vs the abstract signature, while allowing
            for concrete signature to contain additional parameters beyond those defined
            in the abstract signature
        """
        abst_sig = abstract.__signature__
        conc_sig = concrete.__signature__

        sig_mod = _SignatureModifier(concrete)

        # Check for missing parameters in concrete signature
        missing_params = set(abst_sig.parameters) - set(conc_sig.parameters)
        if missing_params:
            raise TypeError(
                f"Missing parameters: {missing_params} from {abstract.name} in "
                f"implementation {concrete.func.__qualname__}"
            )

        # Check that parameter order matches
        for abst_param_name, conc_param_name in zip(
            abst_sig.parameters.keys(), conc_sig.parameters.keys()
        ):
            if abst_param_name != conc_param_name:
                raise TypeError(
                    f'[{concrete.func.__qualname__}] argument "{conc_param_name}" '
                    "does not match name of parameter in abstract function signature"
                )

        # Walk through the parameters (in reverse order to add missing default arg values first),
        # which will not line up because the concrete may have extra parameters
        # Update concrete signature with defaults defined in the abstract signature
        # Verify that extra parameters contain a default value
        conc_params = list(reversed(conc_sig.parameters.values()))
        for conc_param in conc_params:
            conc_param_name = conc_param.name
            if conc_param_name == "resolver" and concrete._include_resolver:
                # Handle "include_resolver" logic; algo should not declare default,
                # but we add a default to make things easier for exact dispatching
                if conc_param.default is not inspect._empty:
                    raise TypeError('"resolver" cannot have a default')
                sig_mod.update_default(None, name=conc_param_name)
            elif conc_param_name not in abst_sig.parameters:
                # Extra concrete params must declare a default value
                if conc_param.default is inspect._empty:
                    raise TypeError(
                        f'[{concrete.func.__qualname__}] argument "{conc_param_name}" is not '
                        "found in abstract signature and must declare a default value"
                    )
            else:
                abst_param = abst_sig.parameters[conc_param_name]
                abst_type = abst_param.annotation

                # Concrete parameters should never define a default value
                # They inherit the default from the abstract signature
                if conc_param.default is not inspect._empty:
                    raise TypeError(
                        f'[{concrete.func.__qualname__}] argument "{conc_param_name}" declares '
                        f"a default value; default values can only be defined in the abstract signature"
                    )
                # If abstract defines a default, update concrete with the same default
                if abst_param.default is not inspect._empty:
                    sig_mod.update_default(abst_param.default, name=conc_param_name)

                # Normalize and check concrete parameter
                conc_type = cls.normalize_concrete_type(
                    resolver,
                    conc_param.annotation,
                    abst_type,
                    sig_mod,
                    name=conc_param_name,
                )
                cls.check_concrete_type(
                    concrete, conc_type, abst_type, name=conc_param_name
                )

        abst_ret = abst_sig.return_annotation
        conc_ret = conc_sig.return_annotation
        # Normalize return type, which might be a tuple
        if typing.get_origin(conc_ret) == tuple:
            if len(typing.get_args(abst_ret)) != len(typing.get_args(conc_ret)):
                raise TypeError(
                    f"{concrete.func.__qualname__} return type is not compatible "
                    "with abstract function signature"
                )
            for index, (conc_ret_sub_type, abst_ret_sub_type) in enumerate(
                zip(typing.get_args(conc_ret), typing.get_args(abst_ret))
            ):
                # Normalize and check concrete return subtype
                conc_ret_sub_type_normalized = cls.normalize_concrete_type(
                    resolver, conc_ret_sub_type, abst_ret_sub_type, sig_mod, index=index
                )
                cls.check_concrete_type(
                    concrete, conc_ret_sub_type_normalized, abst_ret_sub_type
                )
        else:
            # Normalize and check concrete return type
            conc_ret = cls.normalize_concrete_type(
                resolver, conc_ret, abst_ret, sig_mod
            )
            cls.check_concrete_type(concrete, conc_ret, abst_ret)

        return

    @classmethod
    def normalize_concrete_type(
        cls,
        resolver: Resolver,
        conc_type,
        abst_type,
        sig_mod: Union["_SignatureModifier", mgtyping.Combo],
        *,
        name=None,
        index=None,
    ):
        """
        Converts ConcreteType classes into instances.
        For example, NumpyNodeMapType -> NumpyNodeMapType()
        """

        # Convert normal typing objects into Combos
        origin = typing.get_origin(conc_type)
        if origin == Union:
            conc_type = mgtyping.Union[typing.get_args(conc_type)]
            sig_mod.update_annotation(conc_type, name=name, index=index)
        elif origin == list:
            conc_type = mgtyping.List[typing.get_args(conc_type)]
            sig_mod.update_annotation(conc_type, name=name, index=index)

        if type(conc_type) is type and issubclass(conc_type, ConcreteType):
            conc_type = conc_type()
            sig_mod.update_annotation(conc_type, name=name, index=index)
        elif isinstance(conc_type, ConcreteType):
            return conc_type
        elif isinstance(conc_type, mgtyping.Combo):
            if isinstance(sig_mod, mgtyping.Combo):
                # Optional[List] or Optional[Union] is allowed
                if sig_mod.kind is None and sig_mod.optional and not conc_type.optional:
                    sig_mod.kind = conc_type.kind
                    sig_mod.types = conc_type.types
                    conc_type = sig_mod
                else:
                    # Everything else is not allowed
                    raise TypeError(
                        "Nesting a Combo type inside a Combo type is not allowed"
                    )
            # Get the first type from abst_type, which is sufficient for normalization purposes
            # because of the requirement for a common subtype within a Combo
            sub_abst_type = (
                abst_type.types[0] if isinstance(abst_type, mgtyping.Combo) else None
            )
            # Normalize nested types
            for combo_idx, typ_ in enumerate(conc_type.types):
                # conc_type (a Combo) masquerades as sig_mod during this call
                cls.normalize_concrete_type(
                    resolver,
                    typ_,
                    sub_abst_type,
                    sig_mod=conc_type,
                    name=name,
                    index=combo_idx,
                )
            # This must be done after all nested types have been upgraded
            conc_type.compute_common_subtype()
        elif conc_type in resolver.class_to_concrete:
            if isinstance(abst_type, AbstractType):
                conc_type = resolver.class_to_concrete[conc_type]()
                sig_mod.update_annotation(conc_type, name=name, index=index)
            else:
                # If the abstract signatures uses a plain Python object
                # which happens to be a value_type of a ConcreteType, we leave it alone
                pass

        return conc_type

    @classmethod
    def check_concrete_type(
        cls, conc_algo: ConcreteAlgorithm, conc_type, abst_type, *, name=None
    ) -> None:
        """
        This is a helper for normalize_concrete_algorithm_signature.
        This method verifies that an individual pair of abstract and concrete types match.
        """
        msg = (
            conc_algo.func.__qualname__
            + " "
            + (f'argument "{name}"' if name is not None else "return type")
        )

        if isinstance(conc_type, mgtyping.Combo):
            if name is None:
                raise TypeError(f"{msg} may not be a Combo")
            if abst_type.optional != conc_type.optional:
                raise TypeError(
                    f"{msg}: {conc_type} does not match optional flag in {abst_type}"
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
                raise TypeError(f"{msg}: {unmatched_types} not found in {abst_type}")
        elif isinstance(conc_type, ConcreteType):
            if not issubclass(conc_type.abstract, abst_type.__class__):
                raise TypeError(
                    f"{msg} {conc_type} is not a concrete type of {abst_type}"
                )
            if conc_type.abstract_instance is not None:
                raise TypeError(f"{msg} specifies abstract properties")
        else:
            # regular Python types need to match exactly
            if abst_type != conc_type:
                raise TypeError(
                    f"{msg}: {conc_type} does not match {abst_type} in abstract signature"
                )


class _SignatureModifier:
    def __init__(self, algo: Union[AbstractAlgorithm, ConcreteAlgorithm]):
        self.algo = algo

    def update_default(self, new_default, *, name):
        sig = self.algo.__signature__
        newparams = [
            p.replace(default=new_default) if pname == name else p
            for pname, p in sig.parameters.items()
        ]
        self.algo.__signature__ = sig.replace(parameters=newparams)

    def update_annotation(self, new_annotation, *, name=None, index=None):
        if name is None and index is None:  # single return
            return self._update_return_annotation(new_annotation)
        if name is None and index is not None:  # multi return
            return self._update_return_annotation(new_annotation, index=index)
        if name is not None and index is None:  # single arg
            return self._update_arg_annotation(new_annotation, name=name)
        if name is not None and index is not None:  # multi arg
            raise NotImplementedError()

    def _update_arg_annotation(self, new_annotation, name):
        sig = self.algo.__signature__
        newparams = [
            p.replace(annotation=new_annotation) if pname == name else p
            for pname, p in sig.parameters.items()
        ]
        self.algo.__signature__ = sig.replace(parameters=newparams)

    def _update_return_annotation(self, new_return, index=None):
        sig = self.algo.__signature__
        if index is not None:
            assert (
                typing.get_origin(sig.return_annotation) == tuple
            ), "Use of index only supported for `Tuple[]` return type"
            new_rets = (
                new_return if i == index else r
                for i, r in enumerate(typing.get_args(sig.return_annotation))
            )
            # TODO: check if this breaks in Python 3.9+
            sig.return_annotation.__args__ = tuple(new_rets)
        else:
            self.algo.__signature__ = sig.replace(return_annotation=new_return)


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
        self.__signature__ = resolver.abstract_algorithms[algo_name].__signature__
        self.__wrapped__ = abstract_algo

    def __call__(self, *args, **kwargs):
        return self._resolver.run(self._algo_name, *args, **kwargs)

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
