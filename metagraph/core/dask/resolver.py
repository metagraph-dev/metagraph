import types
from dask.base import tokenize
from dask import delayed
from ..resolver import Resolver, PlanNamespace
from .placeholder import Placeholder
from ..plugin import ConcreteType


class DaskResolver:
    _placeholders = {}

    def __init__(self, resolver: Resolver):
        self._resolver = resolver

        # Patch plan namespace
        self.plan = PlanNamespace(self)
        self.plan.algos = self._resolver.plan.algos

        # Add placeholder types to `class_to_concrete`
        self.class_to_concrete = self._resolver.class_to_concrete.copy()
        for ct in self._resolver.concrete_types:
            ph = self._get_placeholder(ct)
            self.class_to_concrete[ph] = ct

    # Default behavior (unless overridden) is to delegate to the original resolver
    def __getattr__(self, item):
        return getattr(self._resolver, item)

    def __dir__(self):
        return dir(self._resolver)

    def _get_placeholder(self, concrete_type):
        if concrete_type not in self._placeholders:
            ph = types.new_class(f"{concrete_type.__name__}Placeholder", (Placeholder,))
            self._placeholders[concrete_type] = ph
        return self._placeholders[concrete_type]

    def _add_translation_plan(self, mst, src, **props):
        obj = src
        src_type = mst.src_type
        for trans, dst_type in zip(mst.translators, mst.dst_types):
            ph = self._get_placeholder(dst_type)
            key = f"translate::{src_type.__name__}->{dst_type.__name__}::{tokenize(ph, trans, obj, props)}"
            if dst_type is mst.final_type:
                obj = ph.build(key, trans, (obj,), props)
            else:
                obj = ph.build(key, trans, (obj,))
            src_type = dst_type
        return obj

    def _add_algorithm_plan(self, algo_plan, *args, **kwargs):
        sig = algo_plan.algo.__signature__
        # Walk through arguments and apply required translations
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, val in bound.arguments.items():
            if name in algo_plan.required_translations:
                trans = algo_plan.required_translations[name]
                bound.arguments[name] = self._add_translation_plan(trans, val)
        args, kwargs = bound.args, bound.kwargs
        # Determine return type and add task
        ret = sig.return_annotation
        if getattr(ret, "__origin__", None) == tuple:
            # TODO: how do we handle multiple return types?
            raise NotImplementedError(
                "Currently can't handle delayed calls which return multiple objects"
            )
        elif type(ret) is not type and isinstance(ret, ConcreteType):
            ct = type(ret)
            ph = self._get_placeholder(ct)
            key = f"algorithm::{algo_plan.algo.abstract_name}::{tokenize(ph, algo_plan, args, kwargs)}"
            return ph.build(key, algo_plan, args, kwargs)
        else:
            # Use dask.delayed instead of a Placeholder
            delayed_call = delayed(algo_plan)
            key = f"algorithm::{algo_plan.algo.abstract_name}::{tokenize(delayed_call, algo_plan, args, kwargs)}"
            return delayed_call(*args, **kwargs, dask_key_name=key)

    def register(self, *args, **kwargs):
        raise NotImplementedError(
            "Register with the resolver prior to creating a DaskResolver"
        )

    def assert_equal(self, *args, **kwargs):
        raise NotImplementedError("Do not assert_equal with a DaskResolver")

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

    def translate(self, value, dst_type, **props):
        translator = self.plan.translate(value, dst_type, **props)
        return self._add_translation_plan(translator, value, **props)
