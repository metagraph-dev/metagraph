from typing import Dict, Optional, Any
from .plugin import ConcreteType
import numpy as np
import scipy.sparse as ss


class MultiStepTranslator:
    def __init__(self, src_type):
        self.src_type = src_type
        self.translators = []
        self.dst_types = []

    def __len__(self):
        return len(self.translators)

    def __iter__(self):
        return iter(self.translators)

    def add_before(self, translator, dst_type):
        self.translators.insert(0, translator)
        self.dst_types.insert(0, dst_type)

    def add_after(self, translator, dst_type):
        self.translators.append(translator)
        self.dst_types.append(dst_type)

    def __call__(self, src, **props):
        if not self.translators:
            return src
        for translator in self.translators[:-1]:
            src = translator(src)
        # Finish by reaching destination along with required properties
        dst = self.translators[-1](src, **props)
        return dst

    def display(self):
        if len(self) == 0:
            print("No translation required")
        if len(self) > 1:
            print("[Multi-step Translation]")
            print(f"(start)  {self.src_type.__name__}")
            for i, nxt_type in enumerate(self.dst_types[:-1]):
                print(f"         {'  ' * i}  -> {nxt_type.__name__}")
            print(f" (end)   {'  ' * (i + 1)}  -> {self.dst_types[-1].__name__}")
        else:
            print("[Direct Translation]")
            print(f"{self.src_type.__name__} -> {self.dst_types[-1].__name__}")

    @classmethod
    def find_translation(
        cls, resolver, src_type, dst_type, *, exact=False
    ) -> Optional["MultiStepTranslator"]:
        if not issubclass(dst_type, ConcreteType):
            dst_type = resolver.class_to_concrete.get(dst_type, dst_type)

        if exact:
            trns = resolver.translators.get((src_type, dst_type), None)
            if trns is None:
                return
            mst = MultiStepTranslator(src_type)
            mst.add_after(trns, dst_type)
            return mst

        abstract = dst_type.abstract
        if abstract not in resolver.translation_matrices:
            # Build translation matrix
            concrete_list = []
            concrete_lookup = {}
            for ct in resolver.concrete_types:
                if issubclass(ct.abstract, abstract):
                    concrete_lookup[ct] = len(concrete_list)
                    concrete_list.append(ct)
            m = ss.dok_matrix((len(concrete_list), len(concrete_list)), dtype=bool)
            for s, d in resolver.translators:
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
            resolver.translation_matrices[abstract] = (
                concrete_list,
                concrete_lookup,
                sssp,
                predecessors,
            )
        # Lookup shortest path from stored results
        packed_data = resolver.translation_matrices[abstract]
        concrete_list, concrete_lookup, sssp, predecessors = packed_data
        try:
            sidx = concrete_lookup[src_type]
            didx = concrete_lookup[dst_type]
        except KeyError:
            return None
        if sssp[sidx, didx] == np.inf:
            return None
        # Path exists; use predecessor matrix to build up required transformations
        mst = MultiStepTranslator(src_type)
        while sidx != didx:
            parent_idx = predecessors[sidx, didx]
            next_translator = resolver.translators[
                (concrete_list[parent_idx], concrete_list[didx])
            ]
            next_dst_type = concrete_list[didx]
            mst.add_before(next_translator, next_dst_type)
            didx = parent_idx

        return mst


class AlgorithmPlan:
    def __init__(
        self, concrete_algorithm, required_translations: Dict[str, MultiStepTranslator]
    ):
        self.algo = concrete_algorithm
        self.required_translations = required_translations

    def __call__(self, *args, **kwargs):
        sig = self.algo.__signature__
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for varname in self.required_translations:
            bound_args.arguments[varname] = self.required_translations[varname](
                bound_args.arguments[varname]
            )
        return self.algo(*bound_args.args, **bound_args.kwargs)

    def display(self):
        sig = self.algo.__signature__
        print(f"{self.algo.__name__}")
        print(f"{self.algo.__signature__}")
        print("=====================")
        print("Argument Translations")
        print("---------------------")
        for varname in sig.parameters:
            if varname in self.required_translations:
                print(f"** {varname} **  ", end="")
                self.required_translations[varname].display()
            else:
                print(f"** {varname} **")
                print(f"{sig.parameters[varname].annotation.__name__}")
        print("---------------------")

    @classmethod
    def build(
        cls, resolver, concrete_algorithm, *args, **kwargs
    ) -> Optional["AlgorithmPlan"]:
        required_translations = {}
        sig = concrete_algorithm.__signature__
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        try:
            parameters = bound_args.signature.parameters
            for arg_name, arg_value in bound_args.arguments.items():
                param_type = parameters[arg_name].annotation
                # If argument type is okay, no need to add an adjustment
                # If argument type is not okay, look for translator
                #   If translator is found, add to required_translations
                #   If no translator is found, return None to indicate failure
                if not cls._check_arg_type(arg_value, param_type):
                    src_type = resolver.typeof(arg_value).__class__
                    translator = MultiStepTranslator.find_translation(
                        resolver, src_type, param_type
                    )
                    if translator is None:
                        return
                    required_translations[arg_name] = translator
            return AlgorithmPlan(concrete_algorithm, required_translations)
        except TypeError:
            return

    @staticmethod
    def _check_arg_type(arg_value, param_type) -> bool:
        if param_type is Any:
            return True
        elif isinstance(param_type, ConcreteType):
            if not param_type.is_satisfied_by_value(arg_value):
                return False
        else:
            if not isinstance(arg_value, param_type):
                return False
        return True
