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
from .planning import MultiStepTranslator, AlgorithmPlan
from .entrypoints import load_plugins
from .typecache import TypeCache, TypeInfo
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


class PlanNamespace:
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
        src_type = self._resolver.typeclass_of(value)
        translator = MultiStepTranslator.find_translation(
            self._resolver, src_type, dst_type
        )
        if translator is None:
            print(
                f"No translation path found for {src_type.__name__} -> {dst_type.__name__}"
            )
        else:
            translator.display()

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
            algo = valid_algos[0]
            return algo.display()

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

        self.typecache = TypeCache()

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

        self.plan = PlanNamespace(self)

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
            # Wipe out existing translation matrices (if any)
            self.translation_matrices = {}

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
                aa = self._normalize_abstract_type(aa)
                self.abstract_algorithms[aa.name] = aa
                self.algo._register(aa.name, Dispatcher(self, aa.name))
                self.plan.algo._register(aa.name, Dispatcher(self.plan, aa.name))

        if concrete_algorithms is not None:
            for ca in concrete_algorithms:
                abstract = self.abstract_algorithms.get(ca.abstract_name, None)
                if abstract is None:
                    raise ValueError(
                        f"concrete algorithm {ca.func.__module__}.{ca.func.__qualname__} implements unregistered abstract algorithm {ca.abstract_name}"
                    )
                self._normalize_concrete_algorithm_signature(abstract, ca)
                self.concrete_algorithms[ca.abstract_name].append(ca)

    def _normalize_abstract_type(self, abst_type):
        if type(abst_type) is type and issubclass(abst_type, AbstractType):
            abst_type = abst_type()
        return abst_type

    def _normalize_concrete_type(self, conc_type, abst_type: AbstractType):
        # handle Python classes used as concrete types
        if abst_type is Any:
            return conc_type
        if isinstance(abst_type, AbstractType) and not isinstance(
            conc_type, ConcreteType
        ):
            if conc_type in self.class_to_concrete:
                return self.class_to_concrete[conc_type]()
            else:
                raise TypeError(
                    f"'{conc_type}' is not a concrete type of '{abst_type}'"
                )
        else:
            return conc_type

    def _normalize_abstract_algorithm_signature(self, abst_algo: AbstractAlgorithm):
        """
        Convert all AbstractType to a no-arg instance
        Leave all Python types alone
        Guard against instances of anything other than AbstractType
        """
        abs_sig = abst_algo.__signature__
        params = abs_sig.parameters
        ret = abs_sig.return_annotation
        changed = False
        for pname, p in params.items():
            if type(p.annotation) == type:
                if issubclass(p.annotation, AbstractType):
                    params[pname] = p.replace(annotation=p.annotation())
                    changed = True
            elif not isinstance(p.annotation, AbstractType):
                raise TypeError(
                    f'{abst_algo.func.__qualname__} argument "{pname}" may not be an instance of type {type(p.annotation)}'
                )
        if type(ret) == type:
            if issubclass(ret, AbstractType):
                ret = ret()
                changed = True
        elif not isinstance(ret, AbstractType):
            raise TypeError(
                f"{abst_algo.func.__qualname__} return type may not be an instance of type {type(ret)}"
            )

        if changed:
            abs_sig = abs_sig.replace(parameters=params, return_annotation=ret)
            abst_algo.__signature__ = abs_sig

    def _normalize_concrete_algorithm_signature(
        self, abstract: AbstractAlgorithm, concrete: ConcreteAlgorithm
    ):
        """
        Convert all ConcreteType to a no-arg instance
        Leave all Python types alone
        Guard against instances of anything other than ConcreteType
        Guard against mismatched signatures vs the abstract signature
        """
        # TODO: update this
        abst_sig = abstract.__signature__
        conc_sig = concrete.__signature__

        # Check parameters
        abst_params = list(abst_sig.parameters.values())
        conc_params = list(conc_sig.parameters.values())
        if len(abst_params) != len(conc_params):
            raise TypeError(
                f"number of parameters does not match between {abstract.func.__qualname__} and {concrete.func.__qualname__}"
            )
        for abst_param, conc_param in zip(abst_params, conc_params):
            abst_type = self._normalize_abstract_type(abst_param.annotation)
            if abst_type is Any:
                continue
            conc_type = self._normalize_concrete_type(
                conc_type=conc_param.annotation, abst_type=abst_type
            )
            if abst_param.name != conc_param.name:
                raise TypeError(
                    f'{concrete.func.__qualname__} argument "{conc_param.name}" does not match name of parameter in abstract function signature'
                )

            if not isinstance(conc_type, ConcreteType):
                # regular Python types need to match exactly
                if abst_type != conc_type:
                    raise TypeError(
                        f'{concrete.func.__qualname__} argument "{conc_param.name}" does not match abstract function signature'
                    )
            else:
                if not issubclass(conc_type.abstract, abst_type.__class__):
                    raise TypeError(
                        f'{concrete.func.__qualname__} argument "{conc_param.name}" does not have type compatible with abstract function signature'
                    )
                if conc_type.abstract_instance is not None:
                    raise TypeError(
                        f'{concrete.func.__qualname__} argument "{conc_param.name}" specifies abstract properties'
                    )
                # If concrete type has specificity limits, make sure they are
                # adequate for the abstract signature's required minimums
                if conc_type.abstract_property_specificity_limits:
                    for key, required_minimum in abst_type.prop_idx.items():
                        if key in conc_type.abstract_property_specificity_limits:
                            max_specificity_str = conc_type.abstract_property_specificity_limits[
                                key
                            ]
                            max_specificity_int = abst_type.properties[key].index(
                                max_specificity_str
                            )
                            if max_specificity_int < required_minimum:
                                raise TypeError(
                                    f'{concrete.func.__qualname__} argument "{key}" has specificity limits which are '
                                    f"incompatible with the abstract signature"
                                )
        abst_ret = self._normalize_abstract_type(abst_sig.return_annotation)
        conc_ret = self._normalize_concrete_type(
            conc_type=conc_sig.return_annotation, abst_type=abst_ret
        )
        if hasattr(conc_ret, "__origin__") and conc_ret.__origin__ == tuple:
            abst_ret_sub_types = abst_ret.__args__
            conc_ret_sub_types = conc_ret.__args__
            if len(abst_ret_sub_types) != len(conc_ret_sub_types):
                raise TypeError(
                    f"{concrete.func.__qualname__} return type is not compatible with abstract function signature"
                )
            for conc_ret_sub_type, abst_ret_sub_type in zip(
                conc_ret_sub_types, abst_ret_sub_types
            ):
                abst_ret_sub_type_normalized = self._normalize_abstract_type(
                    abst_ret_sub_type
                )
                conc_ret_sub_type_normalized = self._normalize_concrete_type(
                    conc_type=conc_ret_sub_type, abst_type=abst_ret_sub_type_normalized
                )
                self._check_concrete_algorithm_return_signature(
                    concrete, conc_ret_sub_type_normalized, abst_ret_sub_type_normalized
                )
        else:
            self._check_concrete_algorithm_return_signature(
                concrete, conc_ret, abst_ret
            )

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
        plugins = load_plugins()
        self.register(**plugins)

    def typeclass_of(self, value):
        """Return the concrete typeclass corresponding to a value"""
        if value in self.typecache:
            typeinfo = self.typecache[value]
            return typeinfo.concrete_typeclass
        else:
            # Check for direct lookup
            concrete_type = self.class_to_concrete.get(type(value))
            if concrete_type is None:
                for ct in self.concrete_types:
                    if ct.is_typeclass_of(value):
                        concrete_type = ct

            if concrete_type is not None:
                typeinfo = TypeInfo(
                    abstract_typeclass=concrete_type.abstract,
                    known_abstract_props={},
                    concrete_typeclass=concrete_type,
                    known_concrete_props={},
                )
                self.typecache[value] = typeinfo
                return concrete_type

        raise TypeError(f"Class {value.__class__} does not have a registered type")

    def type_of(self, value):
        """Return the fully specified type for this value.

        This may require potentially slow computation of properties.  Only use
        this for debugging.
        """
        return self.typeclass_of(value).get_type(value)

    def translate(self, value, dst_type, **props):
        """Convert a value to a new concrete type using translators"""
        src_type = self.typeclass_of(value)
        translator = MultiStepTranslator.find_translation(self, src_type, dst_type)
        if translator is None:
            raise TypeError(f"Cannot convert {value} to {dst_type}")
        return translator(value, **props)

    def find_algorithm_solutions(
        self, algo_name: str, *args, **kwargs
    ) -> List[AlgorithmPlan]:
        if algo_name not in self.abstract_algorithms:
            raise ValueError(f'No abstract algorithm "{algo_name}" has been registered')

        # Find all possible solution paths
        solutions: List[AlgorithmPlan] = []
        for concrete_algo in self.concrete_algorithms.get(algo_name, []):
            plan = AlgorithmPlan.build(self, concrete_algo, *args, **kwargs)
            if plan is None:
                continue

            solutions.append(plan)

        # Sort by fewest number of translations required
        def total_num_translations(plan):
            return sum(len(t) for t in plan.required_translations.values())

        solutions.sort(key=lambda x: total_num_translations(x))

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

    def call_algorithm(self, algo_name: str, *args, **kwargs):
        if algo_name not in self.abstract_algorithms:
            raise ValueError(f'No abstract algorithm "{algo_name}" has been registered')

        # Validate types meeting minimum specificity required by abstract properties
        abstract_algo = self.abstract_algorithms[algo_name]
        sig = abstract_algo.__signature__
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        parameters = bound_args.signature.parameters
        for arg_name, arg_value in bound_args.arguments.items():
            param_type = parameters[arg_name].annotation
            if param_type is Any:
                continue
            if type(param_type) is type:
                if issubclass(param_type, AbstractType):
                    param_type = param_type()
                else:
                    if not isinstance(arg_value, param_type):
                        raise TypeError(
                            f"{arg_name} must be of type {param_type.__name__}, "
                            f"not {type(arg_value).__name__}"
                        )
            if isinstance(param_type, AbstractType):
                this_typeclass = self.typeclass_of(arg_value)
                # The above line should ensure the typeinfo cache is populated
                this_typeinfo = self.typecache[arg_value]

                # Check if arg_value has the right abstract type
                if this_typeclass.abstract != type(param_type):
                    raise TypeError(
                        f"{arg_name} must be of type {type(param_type).__name__}, "
                        f"not {this_typeclass.abstract.__name__}::{this_typeclass.__name__}"
                    )

                # Update cache with required properties
                requested_properties = set(param_type.prop_idx.keys())
                known_properties = this_typeinfo.known_abstract_props
                unknown_properties = set(known_properties.keys()) - requested_properties

                new_properties = this_typeclass.compute_abstract_properties(
                    arg_value, unknown_properties
                )
                known_properties.update(
                    new_properties
                )  # this dict is still in the cache too
                this_abs_type = this_typeclass.abstract(**known_properties)

                unsatisfied_requirements = []
                for abst_prop, min_value in param_type.prop_idx.items():
                    if this_abs_type.prop_idx[abst_prop] < min_value:
                        min_val_obj = param_type.properties[abst_prop][min_value]
                        if type(min_val_obj) is bool:
                            unsatisfied_requirements.append(
                                f" -> `{abst_prop}` must be {min_val_obj}"
                            )
                        else:
                            unsatisfied_requirements.append(
                                f' -> `{abst_prop}` must be at least "{min_val_obj}"'
                            )
                if unsatisfied_requirements:
                    raise ValueError(
                        f'"{arg_name}" with properties\n{this_abs_type.prop_val}\n'
                        + f"does not meet the specificity requirements:\n"
                        + "\n".join(unsatisfied_requirements)
                    )

        valid_algos = self.find_algorithm_solutions(algo_name, *args, **kwargs)
        if not valid_algos:
            raise TypeError(
                f'No concrete algorithm for "{algo_name}" can be satisfied for the given inputs'
            )
        else:
            # choose the solutions requiring the fewest translations
            algo = valid_algos[0]
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
