import typing
from typing import List, Dict, Optional, Any
from .plugin import ConcreteType
from .typing import Combo
from collections import abc
import inspect
import numpy as np
import scipy.sparse as ss
from metagraph import config, Wrapper, NodeID
from .dask.placeholder import Placeholder


class TranslationMatrix:
    def __init__(self, resolver, abstract):
        self.abstract = abstract
        # Note: Do not save `resolver` as an attribute -- `build_mst` may be called with a regular or dask resolver

        # Build translation matrix
        self.concrete_list = concrete_list = []
        self.concrete_lookup = concrete_lookup = {}
        included_abstract_types = set()
        for ct in resolver.concrete_types:
            if (
                abstract is ct.abstract
                or abstract in ct.abstract.unambiguous_subcomponents
            ):
                concrete_lookup[ct] = len(concrete_list)  # index position
                concrete_list.append(ct)
                included_abstract_types.add(ct.abstract)
        m = ss.dok_matrix((len(concrete_list), len(concrete_list)), dtype=bool)
        for s, d in resolver.translators:
            # only accept destinations of included abstract types
            if d.abstract in included_abstract_types:
                sidx = concrete_lookup[s]
                didx = concrete_lookup[d]
                m[sidx, didx] = True
        self.sssp, self.predecessors = ss.csgraph.dijkstra(
            m.tocsr(), return_predecessors=True, unweighted=True
        )

    def build_mst(self, resolver, src_type, dst_type):
        mst = MultiStepTranslator(resolver, src_type, dst_type)
        try:
            sidx = self.concrete_lookup[src_type]
            didx = self.concrete_lookup[dst_type]
        except KeyError:
            mst.unsatisfiable = True
            return mst
        if self.sssp[sidx, didx] == np.inf:
            mst.unsatisfiable = True
            return mst
        # Path exists; use predecessor matrix to build up required transformations
        concrete_list = self.concrete_list
        while sidx != didx:
            parent_idx = self.predecessors[sidx, didx]
            next_translator = resolver.translators[
                (concrete_list[parent_idx], concrete_list[didx])
            ]
            next_dst_type = concrete_list[didx]
            mst.add_before(next_translator, next_dst_type)
            didx = parent_idx

        return mst


class MultiStepTranslator:
    def __init__(self, resolver, src_type, final_type):
        self.resolver = resolver
        self.src_type = src_type
        self.translators = []
        self.dst_types = []
        self.final_type = final_type
        self.unsatisfiable = False

    def __len__(self):
        if self.unsatisfiable:
            raise ValueError(
                f"No translation path found for {self.src_type.__name__} -> {self.final_type.__name__}"
            )
        return len(self.translators)

    def __iter__(self):
        if self.unsatisfiable:
            raise ValueError(
                f"No translation path found for {self.src_type.__name__} -> {self.final_type.__name__}"
            )
        return iter(self.translators)

    def __repr__(self):
        s = []
        if self.unsatisfiable:
            s.append("[Unsatisfiable Translation]")
            s.append(
                f"Translation {self.src_type.__name__} -> {self.final_type.__name__} unsatisfiable"
            )
        elif len(self) == 0:
            s.append("[Null Translation]")
            s.append(
                f"No translation required from {self.src_type.__name__} -> {self.final_type.__name__}"
            )
        elif len(self) > 1:
            s.append("[Multi-step Translation]")
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
                f"No translation path found for {self.src_type.__name__} -> {self.final_type.__name__}"
            )

        if not self.translators:
            return src

        # Import here to avoid circular references
        from .dask.resolver import DaskResolver

        if isinstance(self.resolver, DaskResolver):
            return self.resolver._add_translation_plan(self, src, **props)

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
        dst_type = resolver.class_to_concrete.get(dst_type, dst_type)

        if not isinstance(dst_type, type):
            dst_type = dst_type.__class__

        if exact:
            trns = resolver.translators.get((src_type, dst_type), None)
            mst = MultiStepTranslator(resolver, src_type, dst_type)
            if trns is None:
                mst.unsatisfiable = True
            else:
                mst.add_after(trns, dst_type)
            return mst

        abstract = dst_type.abstract
        if abstract not in resolver._translation_matrices:
            resolver._translation_matrices[abstract] = TranslationMatrix(
                resolver, abstract
            )

        # Lookup shortest path from stored results
        trans_matrix = resolver._translation_matrices[abstract]
        return trans_matrix.build_mst(resolver, src_type, dst_type)


class AlgorithmPlan:
    def __init__(
        self,
        resolver,
        concrete_algorithm,
        required_translations: Dict[str, MultiStepTranslator],
        err_msgs: List[str],
    ):
        self.resolver = resolver
        self.algo = concrete_algorithm
        self.required_translations = required_translations
        self.err_msgs = err_msgs

    @property
    def unsatisfiable(self):
        return len(self.err_msgs) != 0

    @classmethod
    def string_for_annotation(cls, annotation) -> str:
        if type(annotation) in (type, Wrapper):
            result = f"{annotation.__name__}"
        else:
            result = f"{annotation.__class__.__name__}"
        return result

    def __repr__(self):
        sig = self.algo.__signature__
        s = [
            f"{self.algo.__name__}",
            f"{self.algo.__signature__}",
            "=====================",
            "Argument Translations",
            "---------------------",
        ]
        if self.unsatisfiable:
            s += self.err_msgs
        else:
            for varname in sig.parameters:
                if varname in self.required_translations:
                    s.append(f"** {varname} **  {self.required_translations[varname]}")
                else:
                    s.append(f"** {varname} **")
                    anni = sig.parameters[varname].annotation
                    s.append(self.string_for_annotation(anni))
        s.append("---------------------")
        return "\n".join(s)

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs):
        if self.unsatisfiable:
            combined_err_msg = "".join(["\n    " + msg for msg in self.err_msgs])
            raise ValueError(f"Algorithm not callable because: {combined_err_msg}")

        # Import here to avoid circular references
        from .dask.resolver import DaskResolver

        if isinstance(self.resolver, DaskResolver):
            return self.resolver._add_algorithm_plan(self, *args, **kwargs)

        sig = self.algo.__signature__
        # inject resolver into the arguments if concrete algo requested it
        if self.algo._include_resolver:
            kwargs["resolver"] = self.resolver
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for varname, trans in self.required_translations.items():
            if type(trans) is list:
                # This case happens when a Combo List input needs translation
                # Each element of the incoming list is solved for a translator
                # So each element needs to apply its specific translator
                result = [
                    tr(item) for tr, item in zip(trans, bound_args.arguments[varname])
                ]
            else:
                result = trans(bound_args.arguments[varname])
            bound_args.arguments[varname] = result
        return self.algo(*bound_args.args, **bound_args.kwargs)

    def display(self):
        print(self)

    @classmethod
    def build(
        cls, resolver, concrete_algorithm, *args, **kwargs
    ) -> Optional["AlgorithmPlan"]:
        abstract_algo = resolver.abstract_algorithms[concrete_algorithm.abstract_name]
        abstract_params = abstract_algo.__signature__.parameters
        required_translations = {}
        err_msgs = []
        sig = concrete_algorithm.__signature__
        if concrete_algorithm._include_resolver:
            kwargs["resolver"] = resolver
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        try:
            parameters = bound_args.signature.parameters
            for arg_name, arg_value in bound_args.arguments.items():
                # Only compare against listed abstract parameters; assume others are concrete specific
                if arg_name not in abstract_params:
                    continue
                param_type = parameters[arg_name].annotation
                # If argument type is okay, no need to add an adjustment
                # If argument type is not okay, look for translator
                #   If translator is found, add to required_translations
                #   If no translator is found, add message to err_msgs
                translation_param_type = cls._check_arg_needs_translation(
                    resolver, arg_name, arg_value, param_type
                )
                if translation_param_type is not None:
                    failure_message = None
                    if (
                        isinstance(translation_param_type, Combo)
                        and translation_param_type.kind == "List"
                    ):
                        translator = []
                        for iav, av in enumerate(arg_value):
                            src_type = resolver.typeclass_of(av)
                            trans = MultiStepTranslator.find_translation(
                                resolver, src_type, translation_param_type.types[0]
                            )
                            if trans.unsatisfiable:
                                failure_message = f"Failed to find translator to {trans.final_type.__name__} for {arg_name}[{iav}]"
                                break
                            translator.append(trans)
                    else:
                        src_type = resolver.typeclass_of(arg_value)
                        translator = MultiStepTranslator.find_translation(
                            resolver, src_type, translation_param_type
                        )
                        if translator.unsatisfiable:
                            failure_message = f"Failed to find translator to {translator.final_type.__name__} for {arg_name}"
                    if failure_message:
                        err_msgs.append(failure_message)
                        if config.get(
                            "core.planner.build.verbose", False
                        ):  # pragma: no cover
                            print(failure_message)
                    else:
                        required_translations[arg_name] = translator
        except TypeError as e:
            failure_message = f"Failed to find plan due to TypeError:\n{e}"
            err_msgs.append(failure_message)
            if config.get("core.planner.build.verbose", False):  # pragma: no cover
                print(failure_message)
        return AlgorithmPlan(
            resolver, concrete_algorithm, required_translations, err_msgs
        )

    @classmethod
    def _check_arg_needs_translation(cls, resolver, arg_name, arg_value, param_type):
        """
        Returns None if no translation is needed
        If translation is needed, returns the appropriate param_type to build the translation for
        """
        if param_type is Any:
            return
        elif param_type is NodeID:
            if not isinstance(arg_value, int):
                raise TypeError(f"`{arg_name}` Not a valid NodeID: {arg_value}")
            return
        elif isinstance(param_type, ConcreteType):
            arg_typeclass = resolver.typeclass_of(arg_value)

            # Handle lazy objects which don't know their properties
            if isinstance(arg_value, Placeholder):
                if arg_typeclass is not param_type.__class__:
                    return param_type
                else:
                    # TODO: add a self-translation step to ensure correct properties
                    return

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
            elif param_type.kind == "Union":
                if param_type.subtype == "concrete":
                    arg_typeclass = resolver.typeclass_of(arg_value)
                    # Find appropriate abstract type to translate to (don't allow translation between abstract types)
                    for ct in param_type.types:
                        if arg_typeclass.abstract == ct.abstract:
                            return cls._check_arg_needs_translation(
                                resolver, arg_name, arg_value, ct
                            )
                else:
                    for pt in param_type.types:
                        if isinstance(arg_value, pt):
                            return
                raise TypeError(
                    f"`{arg_name}` with type {type(arg_value)} does not match any of {param_type}"
                )
            elif param_type.kind == "List":
                if not isinstance(arg_value, (list, tuple)):
                    raise TypeError(
                        f"`{arg_name}` must be a list, not {type(arg_value)}"
                    )

                item_type = param_type.types[0]
                if isinstance(item_type, ConcreteType):
                    target_types = [
                        cls._check_arg_needs_translation(
                            resolver, arg_name, arg_value_element, item_type
                        )
                        for arg_value_element in arg_value
                    ]
                    unique_target_types = {t for t in target_types if t is not None}
                    if not unique_target_types:
                        # Nothing to translate
                        return
                    num_unique_target_types = len(unique_target_types)
                    if num_unique_target_types == 1:
                        return Combo(list(unique_target_types), kind="List")
                elif all(
                    isinstance(arg_value_element, item_type)
                    for arg_value_element in arg_value
                ):
                    return
                raise TypeError(f"{arg_name}: {arg_value} does not match {param_type}")
            else:  # kind = None
                item_type = param_type.types[0]
                return cls._check_arg_needs_translation(
                    resolver, arg_name, arg_value, item_type
                )
        elif typing.get_origin(param_type) == abc.Callable:
            if not callable(arg_value):
                raise TypeError(f"{arg_name} must be Callable, not {type(arg_value)}")

            required_input_types, required_return_type = typing.get_args(param_type)
            if isinstance(arg_value, np.ufunc):
                if len(required_input_types) != arg_value.nin:
                    raise TypeError(
                        f"number of inputs mismatch: {arg_value.nin} != {len(required_input_types)}"
                    )
            else:
                arg_value_signature = inspect.signature(arg_value)
                actual_input_types = [
                    param.annotation
                    for param in arg_value_signature.parameters.values()
                ]
                if len(required_input_types) != len(actual_input_types):
                    raise TypeError(
                        f"number of inputs mismatch: {len(actual_input_types)} != {len(required_input_types)}"
                    )
                for itype, (actual_type, required_type) in enumerate(
                    zip(actual_input_types, required_input_types)
                ):
                    if actual_type == inspect._empty:
                        # free pass if no type declaration
                        pass
                    elif not issubclass(actual_type, required_type):
                        raise TypeError(
                            f"{arg_name}[{itype}] must be type {required_type}, not {type(actual_type)}"
                        )
                actual_return_type = arg_value_signature.return_annotation
                if actual_return_type == inspect._empty:
                    # free pass if no type declaration
                    pass
                elif not issubclass(actual_return_type, required_return_type):
                    raise TypeError(
                        f"{arg_name} return must be type {required_return_type}, not {type(actual_return_type)}"
                    )
        else:
            if not isinstance(arg_value, param_type):
                return param_type
