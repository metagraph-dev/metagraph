"""A Resolver manages a collection of plugins, resolves types, and dispatches
to concrete algorithms.

"""
from collections import defaultdict
import inspect
from typing import List, Dict, Optional
from .plugin import (
    AbstractType,
    ConcreteType,
    Translator,
    AbstractAlgorithm,
    ConcreteAlgorithm,
)
from .entrypoints import load_plugins


class Namespace:
    """Helper class to construct arbitrary nested namespaces of objects on the fly.

    Objects are registered with their full dotted attribute path, and the appropriate
    nested namespace object structure is automatically constructed as needed.  There
    is no removal mechanism.
    """

    def __init__(self):
        self._attrs = defaultdict(lambda: Namespace())

    def _register(self, path: str, obj):
        parts = path.split(".")
        if len(parts) == 1:
            self._attrs[parts[0]] = obj
        else:
            self._attrs[parts[0]]._register(".".join(parts[1:]), obj)

    def __getattr__(self, name: str):
        if name in self._attrs:
            return self._attrs[name]
        else:
            raise AttributeError(f"'Namespace' object has no attribute '{name}'")

    def __dir__(self):
        return self._attrs.keys()


class Resolver:
    """Manages a collection of plugins (types, translators, and algorithms).

    Provides utilities to resolve the types of objects, select translators,
    and dispatch to concrete algorithms based on type matching.
    """

    def __init__(self):
        self.abstract_types: Set[AbstractType] = set()
        self.concrete_types: Set[ConcreteType] = set()
        self.translators: Dict[Tuple(ConcreteType, ConcreteType), Translator] = {}

        # map abstract name to instance of abstract algorithm
        self.abstract_algorithms: Dict[str, AbstractAlgorithm] = {}

        # map abstract name to list of concrete instances
        self.concrete_algorithms: Dict[str, List[ConcreteAlgorithm]] = defaultdict(list)

        # map python classes to concrete types
        self.class_to_concrete: Dict[type, ConcreteType] = {}

        self.algo = Namespace()

    def register(
        self,
        *,
        abstract_types: Optional[List[AbstractType]] = None,
        concrete_types: Optional[List[ConcreteType]] = None,
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

        if concrete_types is not None:
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

    def find_translator(self, value, dst_type) -> Optional[Translator]:
        src_type = self.typeof(value).__class__
        if not issubclass(dst_type, ConcreteType):
            dst_type = self.class_to_concrete.get(dst_type, dst_type)
        return self.translators.get((src_type, dst_type), None)

    def translate(self, value, dst_type, **props):
        """Convert a value to a new concrete type using translators"""
        translator = self.find_translator(value, dst_type)
        if translator is None:
            raise TypeError(f"Cannot convert {value} to {dst_type}")
        return translator(value, **props)

    def _check_arg_types(self, bound_args: inspect.BoundArguments) -> bool:
        parameters = bound_args.signature.parameters
        for arg_name, arg_value in bound_args.arguments.items():
            param_type = parameters[arg_name].annotation
            if isinstance(param_type, ConcreteType):
                if not param_type.is_satisfied_by_value(arg_value):
                    return False
            else:
                if not isinstance(arg_value, param_type):
                    return False
        return True

    def find_algorithm(
        self, algo_name: str, *args, **kwargs
    ) -> Optional[ConcreteAlgorithm]:
        if algo_name not in self.abstract_algorithms:
            raise ValueError(f'No abstract algorithm "{algo_name}" has been registered')

        for concrete_algo in self.concrete_algorithms.get(algo_name, []):
            sig = concrete_algo.__signature__
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if self._check_arg_types(bound_args):
                return concrete_algo

        return None

    def call_algorithm(self, algo_name: str, *args, **kwargs):
        algo = self.find_algorithm(algo_name, *args, **kwargs)
        if algo is None:
            raise TypeError(
                f'No concrete algorithm for "{algo_name}" can be found matching argument type signature'
            )
        else:
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
