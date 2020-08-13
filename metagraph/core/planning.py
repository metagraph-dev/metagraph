from typing import List, Dict, Optional, Any
from .plugin import ConcreteType
from .typing import Combo
from collections import abc
import inspect
import numpy as np
import scipy.sparse as ss
from metagraph import config, Wrapper, NodeID


class MultiStepTranslator:
    def __init__(self, src_type):
        self.src_type = src_type
        self.translators = []
        self.dst_types = []
        self.unsatisfiable = False

    def __len__(self):
        if self.unsatisfiable:
            raise ValueError(
                "No translation path found for {src_type.__name__} -> {self.dst_types[-1].__name__}"
            )
        return len(self.translators)

    def __iter__(self):
        if self.unsatisfiable:
            raise ValueError(
                "No translation path found for {src_type.__name__} -> {self.dst_types[-1].__name__}"
            )
        return iter(self.translators)

    def __repr__(self):
        s = []
        s.append("[Multi-step Translation]")

        if self.unsatisfiable:
            s.append("Translation unsatisfiable")

        if len(self) == 0:
            s.append("No translation required")

        if len(self) > 1:
            s.append(f"(start)  {self.src_type.__name__}")
            for i, nxt_type in enumerate(self.dst_types[:-1]):
                s.append(f"         {'  ' * i}  -> {nxt_type.__name__}")
            s.append(f" (end)   {'  ' * (i + 1)}  -> {self.dst_types[-1].__name__}")
        else:
            s.append("[Direct Translation]")
            s.append(f"{self.src_type.__name__} -> {self.dst_types[-1].__name__}")
        return "\n".join(s)

    def __str__(self):
        return self.__repr__()

    def add_before(self, translator, dst_type):
        self.translators.insert(0, translator)
        self.dst_types.insert(0, dst_type)

    def add_after(self, translator, dst_type):
        self.translators.append(translator)
        self.dst_types.append(dst_type)

    def __call__(self, src, **props):
        if self.unsatisfiable:
            raise ValueError(
                "No translation path found for {src_type.__name__} -> {self.dst_types[-1].__name__}"
            )

        if not self.translators:
            return src

        if config.get("core.logging.translations"):
            self.display()

        for translator in self.translators[:-1]:
            src = translator(src)
        # Finish by reaching destination along with required properties
        dst = self.translators[-1](src, **props)
        return dst

    def display(self):
        print(self)

    @classmethod
    def find_translation(
        cls, resolver, src_type, dst_type, *, exact=False
    ) -> "MultiStepTranslator":
        if isinstance(dst_type, type) and not issubclass(dst_type, ConcreteType):
            dst_type = resolver.class_to_concrete.get(dst_type, dst_type)

        if not isinstance(dst_type, type):
            dst_type = dst_type.__class__

        if exact:
            trns = resolver.translators.get((src_type, dst_type), None)
            mst = MultiStepTranslator(src_type)
            if trns is None:
                mst.unsatisfiable = True
            else:
                mst.add_after(trns, dst_type)
            return mst

        abstract = dst_type.abstract
        if abstract not in resolver.translation_matrices:
            # Build translation matrix
            concrete_list = []
            concrete_lookup = {}
            included_abstract_types = set()
            for ct in resolver.concrete_types:
                if (
                    abstract is ct.abstract
                    or abstract in ct.abstract.unambiguous_subcomponents
                ):
                    concrete_lookup[ct] = len(concrete_list)
                    concrete_list.append(ct)
                    included_abstract_types.add(ct.abstract)
            m = ss.dok_matrix((len(concrete_list), len(concrete_list)), dtype=bool)
            for s, d in resolver.translators:
                # only accept destinations of included abstract types
                if d.abstract in included_abstract_types:
                    sidx = concrete_lookup[s]
                    didx = concrete_lookup[d]
                    m[sidx, didx] = True
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
        mst = MultiStepTranslator(src_type)
        try:
            sidx = concrete_lookup[src_type]
            didx = concrete_lookup[dst_type]
        except KeyError:
            mst.unsatisfiable = True
            return mst
        if sssp[sidx, didx] == np.inf:
            mst.unsatisfiable = True
            return mst
        # Path exists; use predecessor matrix to build up required transformations
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
        self,
        resolver,
        concrete_algorithm,
        required_translations: Dict[str, MultiStepTranslator],
        build_problem_messages: List[str],
    ):
        self.resolver = resolver
        self.algo = concrete_algorithm
        self.required_translations = required_translations
        self.build_problem_messages = build_problem_messages

    def __repr__(self):
        sig = self.algo.__signature__
        s = [
            f"{self.algo.__name__}",
            f"{self.algo.__signature__}",
            "=====================",
            "Argument Translations",
            "---------------------",
        ]
        if len(self.build_problem_messages) != 0:
            s += self.build_problem_messages
        else:
            for varname in sig.parameters:
                if varname in self.required_translations:
                    s.append(f"** {varname} **  {self.required_translations[varname]}")
                else:
                    s.append(f"** {varname} **")
                    anni = sig.parameters[varname].annotation
                    if type(anni) is Wrapper:
                        s.append(f"{anni.__name__}")
                    elif type(anni) is type:
                        s.append(f"{anni.__name__}")
                    else:
                        s.append(f"{anni.__class__.__name__}")
        s.append("---------------------")
        return "\n".join(s)

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs):
        if len(self.build_problem_messages) != 0:
            conglomerate_build_problem_message = "".join(
                ["\n    " + msg for msg in self.build_problem_messages]
            )
            raise ValueError(
                f"Algorithm not callable because: {conglomerate_build_problem_message}"
            )
        # Defaults are defined in the abstract signature; apply those prior to binding with concrete signature
        args, kwargs = self.apply_abstract_defaults(
            self.resolver, self.algo.abstract_name, *args, **kwargs
        )
        sig = self.algo.__signature__
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for varname in self.required_translations:
            bound_args.arguments[varname] = self.required_translations[varname](
                bound_args.arguments[varname]
            )
        return self.algo(*bound_args.args, **bound_args.kwargs)

    def display(self):
        print(self)

    @classmethod
    def build(
        cls, resolver, concrete_algorithm, *args, **kwargs
    ) -> Optional["AlgorithmPlan"]:
        # Defaults are defined in the abstract signature; apply those prior to binding with concrete signature
        args, kwargs = cls.apply_abstract_defaults(
            resolver, concrete_algorithm.abstract_name, *args, **kwargs
        )
        required_translations = {}
        build_problem_messages = []
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
                #   If no translator is found, add message to build_problem_messages
                translation_param_type = cls._check_arg_type(
                    resolver, arg_name, arg_value, param_type
                )
                if translation_param_type is not None:
                    src_type = resolver.typeclass_of(arg_value)
                    translator = MultiStepTranslator.find_translation(
                        resolver, src_type, translation_param_type
                    )
                    if translator.unsatisfiable:
                        failure_message = f"Failed to find translator for {arg_name}"
                        build_problem_messages.append(failure_message)
                        if config.get("core.planner.build.verbose", False):
                            print(failure_message)
                    else:
                        required_translations[arg_name] = translator
        except TypeError as e:
            failure_message = "Failed to find plan due to TypeError:\n{e}"
            build_problem_messages.append(failure_message)
            if config.get("core.planner.build.verbose", False):
                print(failure_message)
        return AlgorithmPlan(
            resolver, concrete_algorithm, required_translations, build_problem_messages
        )

    @staticmethod
    def _check_arg_type(resolver, arg_name, arg_value, param_type):
        """
        Returns None if no translation is needed
        If translation is needed, returns the appropriate param_type to build the translation for
        """
        if param_type is Any:
            return
        elif param_type is NodeID:
            if not isinstance(arg_value, int):
                raise TypeError(f"{arg_name} Not a valid NodeID: {arg_value}")
            return
        elif isinstance(param_type, ConcreteType):
            arg_typeclass = resolver.typeclass_of(arg_value)

            requested_properties = set(param_type.props.keys())
            properties_dict = arg_typeclass.compute_concrete_properties(
                arg_value, requested_properties
            )
            # Instantiate this with the properties we now know
            arg_type = arg_typeclass(**properties_dict)

            if not param_type.is_satisfied_by(arg_type):
                return param_type
        elif isinstance(param_type, Combo):
            if arg_value is None:
                if not param_type.optional:
                    raise TypeError(f"{arg_name} is not Optional, but None was given")
                return
            elif param_type.strict:
                if param_type.kind == "concrete":
                    arg_typeclass = resolver.typeclass_of(arg_value)
                    # Find appropriate abstract type to translate to (don't allow translation between abstract types)
                    for ct in param_type.types:
                        if arg_typeclass.abstract == ct.abstract:
                            return AlgorithmPlan._check_arg_type(
                                resolver, arg_name, arg_value, ct
                            )
                else:
                    for pt in param_type.types:
                        if isinstance(arg_value, pt):
                            return
                raise TypeError(
                    f"{arg_name} {arg_value} does not match any of {param_type}"
                )
            else:
                # Non-strict should only have a single possible choice
                if len(param_type.types) > 1:
                    raise AssertionError(
                        f"{arg_name} Illegal Combo: non-strict with choice of {param_type.types}"
                    )
                return AlgorithmPlan._check_arg_type(
                    resolver, arg_name, arg_value, list(param_type.types)[0]
                )
        elif getattr(param_type, "__origin__", None) == abc.Callable:
            if not callable(arg_value):
                raise TypeError(f"{arg_name} must be Callable, not {type(arg_value)}")
            # TODO consider using typing.get_type_hints
            arg_value_func_params_desired_types = param_type.__args__[:-1]
            arg_value_func_desired_return_type = param_type.__args__[-1]
            if isinstance(arg_value, np.ufunc):
                if len(arg_value_func_params_desired_types) != arg_value.nin:
                    return param_type
                # TODO use arg_value.nout to compare to arg_value_func_desired_return_type
                if arg_value.signature is not None:
                    pass  # TODO handle this case
            else:
                arg_value_signature = inspect.signature(arg_value)
                arg_value_func_params = arg_value_signature.parameters.values()
                arg_value_func_params_actual_types = (
                    param.annotation for param in arg_value_func_params
                )
                for actual_type, desired_type in zip(
                    arg_value_func_params_actual_types,
                    arg_value_func_params_desired_types,
                ):
                    if (
                        actual_type != inspect._empty
                    ):  # free pass if no type declaration
                        if not issubclass(actual_type, desired_type):
                            return param_type
                arg_value_func_actual_return_type = (
                    arg_value_signature.return_annotation
                )
                if arg_value_func_actual_return_type != inspect._empty:
                    if not issubclass(
                        arg_value_func_actual_return_type,
                        arg_value_func_desired_return_type,
                    ):
                        return param_type
        else:
            if not isinstance(arg_value, param_type):
                return param_type

    @staticmethod
    def apply_abstract_defaults(resolver, algo_name, *args, **kwargs):
        """
        Returns new args and kwargs with defaults applied based on default defined by the abstract algorithm.
        These new args and kwargs are suitable to use when calling concrete algorithms.
        """
        abstract_algo = resolver.abstract_algorithms[algo_name]
        sig = abstract_algo.__signature__
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args.args, bound_args.kwargs
