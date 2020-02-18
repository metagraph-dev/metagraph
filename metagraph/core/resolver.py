"""A Resolver manages a collection of plugins, resolves types, and dispatches
to concrete algorithms.

"""
from collections import defaultdict
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
        self._registered.add(parts[0])
        if len(parts) == 1:
            setattr(self, parts[0], obj)
        else:
            if not hasattr(self, parts[0]):
                setattr(self, parts[0], Namespace())
            getattr(self, parts[0])._register(".".join(parts[1:]), obj)

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
        # Stored result is (concrete_types list, sssp_matrix, predecessors_matrix)
        self.translation_matrices: Dict[
            AbstractType, Tuple[List[ConcreteType], np.ndarray, np.ndarray]
        ] = {}

        self.algo = Namespace()
        self.wrapper = Namespace()

    def register(
        self,
        *,
        abstract_types: Optional[List[AbstractType]] = None,
        concrete_types: Optional[List[ConcreteType]] = None,
        wrappers: Optional[List[Wrapper]] = None,
        translators: Optional[List[Translator]] = None,
        abstract_algorithms: Optional[List[AbstractAlgorithm]] = None,
        concrete_algorithms: Optional[List[ConcreteAlgorithm]] = None,
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
                concrete_types = []
            for wr in wrappers:
                concrete_types.append(wr.Type)
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

        abstract_types = load_plugins("abstract_types")
        concrete_types = load_plugins("concrete_types")
        translators = load_plugins("translators")
        abstract_algorithms = load_plugins("abstract_algorithms")
        concrete_algorithms = load_plugins("concrete_algorithms")

        self.register(
            abstract_types=abstract_types,
            concrete_types=concrete_types,
            translators=translators,
            abstract_algorithms=abstract_algorithms,
            concrete_algorithms=concrete_algorithms,
        )

    def typeof(self, value):
        """Return the concrete type corresponding to a value"""
        # FIXME: silly implementation
        for ct in self.concrete_types:
            try:
                return ct.get_type(value)
            except TypeError:
                pass

        raise TypeError(f"Class {value.__class__} does not have a registered type")

    def find_translation_path(self, src_type, dst_type) -> Optional[Translator]:
        import scipy.sparse as ss

        abstract = dst_type.abstract
        if abstract not in self.translation_matrices:
            # Build translation matrix
            concrete_types = [
                ct for ct in self.concrete_types if issubclass(ct.abstract, abstract)
            ]
            m = ss.dok_matrix((len(concrete_types), len(concrete_types)), dtype=bool)
            for s, d in self.translators:
                try:
                    sidx = concrete_types.index(s)
                    didx = concrete_types.index(d)
                    m[sidx, didx] = True
                except ValueError:
                    pass
            sssp, predecessors = ss.csgraph.dijkstra(
                m.tocsr(), return_predecessors=True, unweighted=True
            )
            self.translation_matrices[abstract] = (concrete_types, sssp, predecessors)
        # Lookup shortest path from stored results
        concrete_types, sssp, predecessors = self.translation_matrices[abstract]
        sidx = concrete_types.index(src_type)
        didx = concrete_types.index(dst_type)
        if sssp[sidx, didx] == np.inf:
            return None
        # Path exists; use predecessor matrix to build up required transformations
        steps = []
        while True:
            parent_idx = predecessors[sidx, didx]
            steps.insert(
                0, self.translators[(concrete_types[parent_idx], concrete_types[didx])]
            )
            if parent_idx == sidx:
                break
            didx = parent_idx

        def walk_steps(src, **props):
            # print(f'Walking to final translation. Starting type = {type(src)}')
            # Move source along shortest path
            for trns in steps[:-1]:
                src = trns(src)
                # print(f'Walking to final translation. Intermediate type = {type(src)}')
            # Finish by reaching destination along with required properties
            dst = steps[-1](src, **props)
            # print(f'Walking to final translation. Final type = {type(dst)}')
            return dst

        trns = Translator(walk_steps)
        trns._num_steps = len(
            steps
        )  # monkey patch to pass along information about complexity
        return trns

    def find_translator(self, value, dest_type, *, exact=False) -> Optional[Translator]:
        src_type = self.typeof(value).__class__
        dst_type = dest_type  # save original for error message
        if not issubclass(dst_type, ConcreteType):
            dst_type = self.class_to_concrete.get(dst_type, dst_type)
        ret_val = self.translators.get((src_type, dst_type), None)
        if ret_val is None and not exact and src_type != dst_type:
            ret_val = self.find_translation_path(src_type, dst_type)
            if ret_val is None:
                # errmsg = (
                #     f'No translator found for {src_type} -> {dst_type}\n'
                #     f'incoming type: {type(value)}\n'
                #     f'outgoing type: {dest_type}'
                # )
                # raise ResolveFailureError(errmsg)
                return None
        return ret_val

    def translate(self, value, dst_type, **props):
        """Convert a value to a new concrete type using translators"""
        translator = self.find_translator(value, dst_type)
        if translator is None:
            raise TypeError(f"Cannot convert {value} to {dst_type}")
        return translator(value, **props)

    def _find_required_translations(
        self, bound_args: inspect.BoundArguments
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
                    translator = self.find_translator(arg_value, param_type)
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

        # Find all possible solution paths
        solutions: List[
            Tuple[int, ConcreteAlgorithm]
        ] = []  # ranked by number of translations required
        for concrete_algo in self.concrete_algorithms.get(algo_name, []):
            sig = concrete_algo.__signature__
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            required_translations = self._find_required_translations(bound_args)
            if required_translations is None:  # No solution found
                continue
            elif not required_translations:  # Exact solution found
                solutions.append((0, concrete_algo))
            else:  # Solution found which requires translations
                num_translations = 0
                for trns in required_translations.values():
                    # On-the-fly translators have been monkey-patched; original translators have not
                    num_translations += getattr(trns, "_num_steps", 1)

                def onthefly_concrete_algo(*args, **kwargs):
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    for varname in required_translations:
                        # print(f'applying adjustment to {varname}. starting type = {type(bound_args.arguments[varname])}')
                        bound_args.arguments[varname] = required_translations[varname](
                            bound_args.arguments[varname]
                        )
                        # print(f'applying adjustment to {varname}. ending type = {type(bound_args.arguments[varname])}')
                    return concrete_algo(*bound_args.args, **bound_args.kwargs)

                solutions.append((num_translations, onthefly_concrete_algo))

        solutions.sort()
        return solutions

    def find_algorithm(
        self, algo_name: str, *args, **kwargs
    ) -> Optional[ConcreteAlgorithm]:
        valid_algos = self.find_algorithm_solutions(algo_name, *args, **kwargs)
        if not valid_algos:
            return None
        return valid_algos[0][-1]

    def call_algorithm(self, algo_name: str, *args, **kwargs):
        valid_algos = self.find_algorithm_solutions(algo_name, *args, **kwargs)
        if not valid_algos:
            raise TypeError(
                f'No concrete algorithm for "{algo_name}" can be found matching argument type signature'
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
