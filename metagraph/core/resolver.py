"""A Resolver manages a collection of plugins, resolves types, and dispatches
to concrete algorithms.

"""
from collections import defaultdict
from functools import partial
import inspect
from typing import List, Tuple, Set, Dict, Callable, Optional, Any
from .plugin import (
    AbstractType,
    ConcreteType,
    Wrapper,
    Translator,
    AbstractAlgorithm,
    ConcreteAlgorithm,
)
from .entrypoints import load_plugins
import numpy as np


class ResolveFailureError(Exception):
    pass


class NamespaceError(Exception):
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

        # map abstract name to list of concrete instances
        self.concrete_algorithms: Dict[str, List[ConcreteAlgorithm]] = defaultdict(list)

        # map python classes to concrete types
        self.class_to_concrete: Dict[type, ConcreteType] = {}

        # translation graph matrices
        # Single-sourch shortest path matrix and predecessor matrix from scipy.sparse.csgraph.dijkstra
        # Stored result is (concrete_types list, concrete_types lookup, sssp_matrix, predecessors_matrix)
        self.translation_matrices: Dict[
            AbstractType,
            Tuple[List[ConcreteType], Dict[ConcreteType, int], np.ndarray, np.ndarray],
        ] = {}

        self.algo = Namespace()
        self.wrapper = Namespace()
        self.types = Namespace()

        self.plan = Planner(self)

    def register(
        self,
        *,
        abstract_types: Optional[Set[AbstractType]] = None,
        concrete_types: Optional[Set[ConcreteType]] = None,
        wrappers: Optional[Set[Wrapper]] = None,
        translators: Optional[Set[Translator]] = None,
        abstract_algorithms: Optional[Set[AbstractAlgorithm]] = None,
        concrete_algorithms: Optional[Set[ConcreteAlgorithm]] = None,
    ):
        """Register plugins for use with this resolver.

        Plugins will be processed in category order (see function signature)
        to ensure that abstract types are registered before concrete types,
        concrete types before translators, and so on.

        This function may be called multiple times to add additional plugins
        at any time.  Plugins cannot be removed.
        """

        if abstract_types is not None:
            for at in abstract_types:
                if at in self.abstract_types:
                    name = at.__qualname__
                    raise ValueError(f"abstract type {name} already exists")
                self.abstract_types.add(at)

        if wrappers is not None:
            # Let concrete type associated with each wrapper be handled by concrete_types list
            if concrete_types is None:
                concrete_types = set()
            else:
                concrete_types = set(concrete_types)  # copy; don't mutate the original
            for wr in wrappers:
                concrete_types.add(wr.Type)
                # Make wrappers available via resolver.wrappers.<abstract name>.<wrapper name>
                path = f"{wr.Type.abstract.__name__}.{wr.__name__}"
                self.wrapper._register(path, wr)

        if concrete_types is not None:
            self.translation_matrices.clear()  # Force a rebuild with new concrete types
            for ct in concrete_types:
                name = ct.__qualname__
                # ct.abstract cannot be None due to ConcreteType.__init_subclass__
                if ct.abstract not in self.abstract_types:
                    abstract_name = ct.abstract.__qualname__
                    raise ValueError(
                        f"concrete type {name} has unregistered abstract type {abstract_name}"
                    )
                if ct.value_type in self.class_to_concrete:
                    raise ValueError(
                        f"Python class '{ct.value_type}' already has a registered concrete type: {self.class_to_concrete[ct.value_type]}"
                    )

                self.concrete_types.add(ct)
                if ct.value_type is not None:
                    self.class_to_concrete[ct.value_type] = ct

                # Make types available via resolver.types.<abstract name>.<concrete name>
                path = f"{ct.abstract.__name__}.{ct.__name__}"
                self.types._register(path, ct)

        if translators is not None:
            for tr in translators:
                signature = inspect.signature(tr.func)
                src_type = next(iter(signature.parameters.values())).annotation
                src_type = self.class_to_concrete.get(src_type, src_type)
                dst_type = signature.return_annotation
                dst_type = self.class_to_concrete.get(dst_type, dst_type)
                if src_type.abstract != dst_type.abstract:
                    raise ValueError(
                        f"Translator {tr.__class__.__qualname__} must convert between concrete types of same abstract type"
                    )

                self.translators[(src_type, dst_type)] = tr

        if abstract_algorithms is not None:
            for aa in abstract_algorithms:
                if aa.name in self.abstract_algorithms:
                    raise ValueError(f"abstract algorithm {aa.name} already exists")
                self.abstract_algorithms[aa.name] = aa
                self.algo._register(aa.name, Dispatcher(self, aa.name))
                self.plan.algo._register(aa.name, Dispatcher(self.plan, aa.name))

        if concrete_algorithms is not None:
            for ca in concrete_algorithms:
                abstract = self.abstract_algorithms.get(ca.abstract_name, None)
                if abstract is None:
                    raise ValueError(
                        f"concrete algorithm {ca.__class__.__qualname__} implements unregistered abstract algorithm {ca.abstract_name}"
                    )
                abstract = self.abstract_algorithms[ca.abstract_name]
                self._raise_if_concrete_algorithm_signature_invalid(abstract, ca)
                self.concrete_algorithms[ca.abstract_name].append(ca)

    def _normalize_conc_type(self, conc_type, abs_type: AbstractType):
        # handle Python classes used as concrete types
        if abs_type is Any:
            return conc_type
        if issubclass(abs_type, AbstractType) and not isinstance(
            conc_type, ConcreteType
        ):
            if conc_type in self.class_to_concrete:
                return self.class_to_concrete[conc_type]()
            else:
                raise TypeError(f"'{conc_type}' is not a concrete type of '{abs_type}'")
        else:
            return conc_type

    def _raise_if_concrete_algorithm_signature_invalid(
        self, abstract: AbstractAlgorithm, concrete: ConcreteAlgorithm
    ):
        abs_sig = abstract.__signature__
        conc_sig = concrete.__signature__

        # Check parameters
        abs_params = list(abs_sig.parameters.values())
        conc_params = list(conc_sig.parameters.values())
        if len(abs_params) != len(conc_params):
            raise TypeError(
                f"number of parameters does not match between {abstract.func.__qualname__} and {concrete.func.__qualname__}"
            )
        for abs_param, conc_param in zip(abs_params, conc_params):
            abs_type = abs_param.annotation
            conc_type = self._normalize_conc_type(
                conc_type=conc_param.annotation, abs_type=abs_type
            )

            if abs_param.name != conc_param.name:
                raise TypeError(
                    f'{concrete.func.__qualname__} argument "{conc_param.name}" does not match name of parameter in abstract function signature'
                )

            if not isinstance(conc_type, ConcreteType):
                # regular Python types need to match exactly
                if abs_type != conc_type:
                    raise TypeError(
                        f'{concrete.func.__qualname__} argument "{conc_param.name}" does not match abstract function signature'
                    )
            else:
                if not issubclass(conc_type.abstract, abs_type):
                    raise TypeError(
                        f'{concrete.func.__qualname__} argument "{conc_param.name}" does not have type compatible with abstract function signature'
                    )

        # Check return type
        abs_ret = abs_sig.return_annotation
        conc_ret = self._normalize_conc_type(
            conc_type=conc_sig.return_annotation, abs_type=abs_ret
        )
        if not isinstance(conc_ret, ConcreteType):
            # regular Python types need to match exactly
            if abs_ret != conc_ret:
                raise TypeError(
                    f"{concrete.func.__qualname__} return type does not match abstract function signature"
                )
        else:
            if not issubclass(conc_ret.abstract, abs_ret):
                raise TypeError(
                    f"{concrete.func.__qualname__} return type is not compatible with abstract function signature"
                )

    def load_plugins_from_environment(self):
        """Scans environment for plugins and populates registry with them."""
        plugins = load_plugins()
        self.register(**plugins)

    def typeof(self, value):
        """Return the concrete type corresponding to a value"""
        concrete_type = self.class_to_concrete.get(type(value))
        if concrete_type is not None:
            return concrete_type.get_type(value)

        for ct in self.concrete_types:
            try:
                return ct.get_type(value)
            except TypeError:
                pass

        raise TypeError(f"Class {value.__class__} does not have a registered type")

    def find_translation_path(
        self, src_type, dst_type, planning_only=False
    ) -> Optional[Translator]:
        import scipy.sparse as ss

        abstract = dst_type.abstract
        if abstract not in self.translation_matrices:
            # Build translation matrix
            concrete_list = []
            concrete_lookup = {}
            for ct in self.concrete_types:
                if issubclass(ct.abstract, abstract):
                    concrete_lookup[ct] = len(concrete_list)
                    concrete_list.append(ct)
            m = ss.dok_matrix((len(concrete_list), len(concrete_list)), dtype=bool)
            for s, d in self.translators:
                # only accept destinations of specific abstract type
                if d.abstract == abstract:
                    try:
                        sidx = concrete_lookup[s]
                        didx = concrete_lookup[d]
                        m[sidx, didx] = True
                    except KeyError:
                        pass
            sssp, predecessors = ss.csgraph.dijkstra(
                m.tocsr(), return_predecessors=True, unweighted=True
            )
            self.translation_matrices[abstract] = (
                concrete_list,
                concrete_lookup,
                sssp,
                predecessors,
            )
        # Lookup shortest path from stored results
        packed_data = self.translation_matrices[abstract]
        concrete_list, concrete_lookup, sssp, predecessors = packed_data
        try:
            sidx = concrete_lookup[src_type]
            didx = concrete_lookup[dst_type]
        except KeyError:
            if planning_only:
                raise
            else:
                return None
        if sssp[sidx, didx] == np.inf:
            return None
        # Path exists; use predecessor matrix to build up required transformations
        steps = []
        while True:
            parent_idx = predecessors[sidx, didx]
            if planning_only:
                steps.insert(0, concrete_list[didx])
            else:
                steps.insert(
                    0,
                    self.translators[(concrete_list[parent_idx], concrete_list[didx])],
                )
            if parent_idx == sidx:
                break
            didx = parent_idx

        if planning_only:

            def walk_steps(src, **props):
                if len(steps) > 1:
                    print("[Multi-step Translation]")
                    print(f"(start)  {type(src).__name__}")
                    for i, src_type in enumerate(steps[:-1]):
                        print(f"         {'  '*i}  -> {src_type.__name__}")
                    print(f" (end)   {'  '*(i+1)}  -> {dst_type.__name__}")
                else:
                    print("[Direct Translation]")
                    print(f"{type(src).__name__} -> {dst_type.__name__}")

        else:

            def walk_steps(src, **props):
                # Move source along shortest path
                for trns in steps[:-1]:
                    src = trns(src)
                # Finish by reaching destination along with required properties
                dst = steps[-1](src, **props)
                return dst

        trns = Translator(walk_steps)
        trns._num_steps = len(
            steps
        )  # monkey patch to pass along information about complexity
        return trns

    def find_translator(
        self, value, dst_type, *, exact=False, planning_only=False
    ) -> Optional[Translator]:
        src_type = self.typeof(value).__class__
        if not issubclass(dst_type, ConcreteType):
            dst_type = self.class_to_concrete.get(dst_type, dst_type)

        if planning_only:
            if exact:
                raise ValueError(
                    "Set `exact` or `planning_only`, but not both when calling `find_translator`"
                )
            ret_val = self.find_translation_path(src_type, dst_type, planning_only=True)
            if ret_val is None:
                raise ResolveFailureError(
                    f"No translation path found for {src_type.__name__} -> {dst_type.__name__}"
                )
            return ret_val

        ret_val = self.translators.get((src_type, dst_type), None)
        if ret_val is None and not exact and src_type != dst_type:
            ret_val = self.find_translation_path(src_type, dst_type)
        return ret_val

    def translate(self, value, dst_type, **props):
        """Convert a value to a new concrete type using translators"""
        translator = self.find_translator(value, dst_type)
        if translator is None:
            raise TypeError(f"Cannot convert {value} to {dst_type}")
        return translator(value, **props)

    def _find_required_translations(
        self, bound_args: inspect.BoundArguments, planning_only=False
    ) -> Optional[Dict[str, Translator]]:
        """
        Attempts to find translators required to make arguments compatible
        If successful, returns a dict of required translations
        If unsuccessful, returns None
        """
        required_translations = {}
        try:
            parameters = bound_args.signature.parameters
            for arg_name, arg_value in bound_args.arguments.items():
                param_type = parameters[arg_name].annotation
                # If argument type is okay, no need to add an adjustment
                # If argument type is not okay, look for translator
                #   If translator is found, add to required_translations
                #   If no translator is found, return None to indicate failure
                if not self._check_arg_type(arg_value, param_type):
                    translator = self.find_translator(
                        arg_value, param_type, planning_only=planning_only
                    )
                    if translator is None:
                        return
                    required_translations[arg_name] = translator
            return required_translations
        except TypeError:
            return

    def _check_arg_type(self, arg_value, param_type) -> bool:
        if param_type is Any:
            return True
        elif isinstance(param_type, ConcreteType):
            if not param_type.is_satisfied_by_value(arg_value):
                return False
        else:
            if not isinstance(arg_value, param_type):
                return False
        return True

    def find_algorithm_solutions(
        self, algo_name: str, *args, **kwargs
    ) -> List[Tuple[int, ConcreteAlgorithm]]:
        if algo_name not in self.abstract_algorithms:
            raise ValueError(f'No abstract algorithm "{algo_name}" has been registered')

        planning_only = kwargs.pop("__mg__planning_only", False)

        # Find all possible solution paths
        solutions: List[
            Tuple[int, ConcreteAlgorithm]
        ] = []  # ranked by number of translations required
        for concrete_algo in self.concrete_algorithms.get(algo_name, []):
            sig = concrete_algo.__signature__
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            try:
                required_translations = self._find_required_translations(
                    bound_args, planning_only=planning_only
                )
                if required_translations is None:  # No solution found
                    continue
            except ResolveFailureError:  # No solution found in planning_only mode
                continue

            num_translations = 0
            for trns in required_translations.values():
                # On-the-fly translators have been monkey-patched; original translators have not
                num_translations += getattr(trns, "_num_steps", 1)

            if planning_only:

                def planning_steps(
                    concrete_algo, required_translations, *args, **kwargs
                ):
                    sig = concrete_algo.__signature__
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    parameters = bound_args.signature.parameters
                    print(f"{concrete_algo.__name__}")
                    print(f"{concrete_algo.__signature__}")
                    print("=====================")
                    print("Argument Translations")
                    print("---------------------")
                    for varname in bound_args.arguments:
                        if varname in required_translations:
                            print(f"** {varname} **  ", end="")
                            bound_args.arguments[varname] = required_translations[
                                varname
                            ](bound_args.arguments[varname])
                        else:
                            print(f"** {varname} **  [No Translation Required]")
                            print(f"{parameters[varname].annotation.__name__}")
                    print("---------------------")

                func = partial(planning_steps, concrete_algo, required_translations)

            elif not required_translations:  # Exact solution found
                func = concrete_algo

            else:  # Solution found which requires translations

                def onthefly_concrete_algo(
                    concrete_algo, required_translations, *args, **kwargs
                ):
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    for varname in required_translations:
                        bound_args.arguments[varname] = required_translations[varname](
                            bound_args.arguments[varname]
                        )
                    return concrete_algo(*bound_args.args, **bound_args.kwargs)

                func = partial(
                    onthefly_concrete_algo, concrete_algo, required_translations
                )

            solutions.append((num_translations, func))

        solutions.sort(key=lambda x: x[0])
        return solutions

    def find_algorithm_exact(
        self, algo_name: str, *args, **kwargs
    ) -> Optional[ConcreteAlgorithm]:
        valid_algos = self.find_algorithm_solutions(algo_name, *args, **kwargs)
        if valid_algos:
            num_translations, best_algo = valid_algos[0]
            if num_translations == 0:
                return best_algo

    def find_algorithm(
        self, algo_name: str, *args, **kwargs
    ) -> Optional[ConcreteAlgorithm]:
        valid_algos = self.find_algorithm_solutions(algo_name, *args, **kwargs)
        if valid_algos:
            num_translations, best_algo = valid_algos[0]
            return best_algo

    def call_algorithm(self, algo_name: str, *args, **kwargs):
        valid_algos = self.find_algorithm_solutions(algo_name, *args, **kwargs)
        if not valid_algos:
            raise TypeError(
                f'No concrete algorithm for "{algo_name}" can be satisfied for the given inputs'
            )
        else:
            # choose the solutions requiring the fewest translations
            algo = valid_algos[0][-1]
            return algo(*args, **kwargs)


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
            print(f"\t{ca.func.__annotations__}")


class Planner:
    """
    Mimics the resolver, but instead of performing real work, it prints the steps that would
    be taken by the resolver when translating or calling algorithms
    """

    def __init__(self, resolver):
        self._resolver = resolver
        self.algo = Namespace()

    def translate(self, value, dst_type, **props):
        """
        Print the steps taken to go from type of value to dst_type
        """
        try:
            translator = self._resolver.find_translator(
                value, dst_type, planning_only=True
            )
            translator(value, **props)
        except ResolveFailureError as e:
            print(e)

    def call_algorithm(self, algo_name: str, *args, **kwargs):
        valid_algos = self._resolver.find_algorithm_solutions(
            algo_name, *args, __mg__planning_only=True, **kwargs
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
            algo = valid_algos[0][-1]
            return algo(*args, **kwargs)

    @property
    def abstract_algorithms(self):
        return self._resolver.abstract_algorithms
