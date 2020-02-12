"""A resolver manages the linkage between plugins, and match values to types"""
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


class Resolver:
    def __init__(self):
        self.abstract_types: Set[AbstractType] = set()
        self.concrete_types: Set[ConcreteType] = set()
        self.translators: Dict[Tuple(ConcreteType, ConcreteType), Translator] = {}

        # map abstract name to instance of abstract algorithm
        self.abstract_algorithms: Dict[str, AbstractAlgorithm] = {}

        # map abstract name to list of concrete instances
        self.concrete_algorithms: Dict[str, List[ConcreteAlgorithm]] = defaultdict(list)

    def register(
        self,
        *,
        abstract_types: Optional[List[AbstractType]] = None,
        concrete_types: Optional[List[ConcreteType]] = None,
        translators: Optional[List[Translator]] = None,
        abstract_algorithms: Optional[List[AbstractAlgorithm]] = None,
        concrete_algorithms: Optional[List[ConcreteAlgorithm]] = None,
    ):

        if abstract_types is not None:
            for at in abstract_types:
                if at in self.abstract_types:
                    name = at.__qualname__
                    raise ValueError(f"abstract type {name} already exists")
                self.abstract_types.add(at)

        if concrete_types is not None:
            for ct in concrete_types:
                name = ct.__qualname__
                if ct.abstract is None:
                    raise ValueError(
                        f"concrete type {name} does not have an abstract type"
                    )
                if ct.abstract not in self.abstract_types:
                    abstract_name = ct.abstract.__qualname__
                    raise ValueError(
                        f"concrete type {name} has unregistered abstract type {abstract_name}"
                    )
                self.concrete_types.add(ct)

        if translators is not None:
            for tr in translators:
                signature = inspect.signature(tr.func)
                src_type = next(iter(signature.parameters.values())).annotation
                dst_type = signature.return_annotation
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

    @staticmethod
    def _raise_if_concrete_algorithm_signature_invalid(abstract, concrete):
        abs_sig = abstract.get_signature()
        conc_sig = concrete.get_signature()

        # Check parameters
        abs_params = list(abs_sig.parameters.values())
        conc_params = list(conc_sig.parameters.values())
        if len(abs_params) != len(conc_params):
            raise TypeError(
                f"number of parameters does not match between {abstract.func.__qualname__} and {concrete.func.__qualname__}"
            )
        for abs_param, conc_param in zip(abs_params, conc_params):
            abs_type = abs_param.annotation
            conc_type = conc_param.annotation

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
        conc_ret = conc_sig.return_annotation
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

    def find_translator(self, value, dst_type):
        src_type = self.typeof(value).__class__
        return self.translators.get((src_type, dst_type), None)

    def translate(self, value, dst_type, **props):
        """Convert a value to a new concrete type using translators"""
        translator = self.find_translator(value, dst_type)
        if translator is None:
            raise TypeError(f"Cannot convert {value} to {dst_type}")
        return translator(value, **props)

    @staticmethod
    def _check_arg_types(bound_args):
        parameters = bound_args.signature.parameters
        for arg_name, arg_value in bound_args.arguments.items():
            if not parameters[arg_name].annotation.is_satisfied_by_value(arg_value):
                return False
        return True

    def find_algorithm(self, algo_name, *args, **kwargs):
        if algo_name not in self.abstract_algorithms:
            raise ValueError(f'No abstract algorithm "{algo_name}" has been registered')

        for concrete_algo in self.concrete_algorithms.get(algo_name, []):
            sig = concrete_algo.get_signature()
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if self._check_arg_types(bound_args):
                return concrete_algo

        return None

    def call_algorithm(self, algo_name, *args, **kwargs):
        algo = self.find_algorithm(algo_name, *args, **kwargs)
        if algo is None:
            raise TypeError(
                f'No concrete algorithm for "{algo_name}" can be found matching argument type signature'
            )
        else:
            return algo(*args, **kwargs)
