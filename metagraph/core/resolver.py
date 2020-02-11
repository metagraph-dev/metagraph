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
                # FIXME: type check here
                if ca.abstract_name not in self.abstract_algorithms:
                    raise ValueError(
                        f"concrete algorithm {ca.__class__.__qualname__} implements unregistered abstract algorithm {ca.abstract_name}"
                    )
                self.concrete_algorithms[ca.abstract_name].append(ca)

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
