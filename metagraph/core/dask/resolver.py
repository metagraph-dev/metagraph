import types
from dask.base import tokenize
from dask import delayed, is_dask_collection
from ..resolver import Resolver, Namespace, PlanNamespace, Dispatcher
from .placeholder import Placeholder, DelayedWrapper
from ..plugin import ConcreteType, MetaWrapper
from typing import Optional


class DaskResolver:
    _placeholders = {}

    def __init__(self, resolver: Resolver):
        self._resolver = resolver

        # Copy class_to_concrete (will be added to further down)
        self.class_to_concrete = self._resolver.class_to_concrete.copy()

        # Patch plan namespace
        self.plan = PlanNamespace(self)
        self.plan.algos = self._resolver.plan.algos

        # Patch algorithms
        def build_algos(namespace):
            for name in dir(namespace):
                obj = getattr(namespace, name)
                if isinstance(obj, Dispatcher):
                    self.algos._register(
                        obj._algo_name, Dispatcher(self, obj._algo_name)
                    )
                elif isinstance(obj, Namespace):
                    build_algos(obj)

        self.algos = Namespace()
        build_algos(self._resolver.algos)

        # Patch wrappers
        def build_wrappers(namespace):
            for name in dir(namespace):
                obj = getattr(namespace, name)
                if isinstance(obj, MetaWrapper):
                    dwrap = self.delayed_wrapper(obj, obj.Type)
                    self.wrappers._register(
                        f"{obj.Type.abstract.__name__}.{obj.__name__}", dwrap,
                    )
                    self.class_to_concrete[dwrap] = dwrap.Type
                elif isinstance(obj, Namespace):
                    build_wrappers(obj)

        self.wrappers = Namespace()
        build_wrappers(self._resolver.wrappers)

        # Add placeholder types to `class_to_concrete`
        for ct in self._resolver.concrete_types:
            ph = self._get_placeholder(ct)
            self.class_to_concrete[ph] = ct

    # Default behavior (unless overridden) is to delegate to the original resolver
    def __getattr__(self, item):
        obj = getattr(self._resolver, item)
        if type(obj) is types.MethodType and type(obj.__self__) is not type:
            # Replace original resolver with the dask resolver for instance methods
            return obj.__func__.__get__(self)
        return obj

    def __dir__(self):
        return dir(self._resolver)

    def delayed_wrapper(self, klass, concrete_type: Optional[ConcreteType] = None):
        ct = concrete_type
        if ct is None:
            ct = self._resolver.class_to_concrete.get(klass)
            if ct is None:
                raise TypeError(
                    f"{klass.__name__} is not a defined `value_type`. Must provide `concrete_type`."
                )
        ph = self._get_placeholder(ct)
        return DelayedWrapper(klass, ph)

    def _get_placeholder(self, concrete_type):
        if concrete_type not in self._placeholders:
            ph = types.new_class(f"{concrete_type.__name__}Placeholder", (Placeholder,))
            ph.concrete_type = concrete_type
            self._placeholders[concrete_type] = ph
        return self._placeholders[concrete_type]

    def _add_translation_plan(self, mst, src, **props):
        obj = src
        src_type = mst.src_type
        for trans, dst_type in zip(mst.translators, mst.dst_types):
            ph = self._get_placeholder(dst_type)
            key = (
                f"translate-{tokenize(ph, trans, obj, props)}",
                f"{src_type.__name__}->{dst_type.__name__}",
            )
            kwargs = {}
            # Pass in the real resolver if needed by the translator
            if trans._include_resolver:
                kwargs["resolver"] = self._resolver
            if dst_type is mst.final_type:
                kwargs.update(props)
            obj = ph.build(key, trans, (obj,), kwargs)
            src_type = dst_type
        return obj

    def _add_algorithm_plan(self, algo_plan, *args, **kwargs):
        args, kwargs = algo_plan.apply_abstract_defaults(
            self._resolver, algo_plan.algo.abstract_name, *args, **kwargs
        )
        sig = algo_plan.algo.__signature__
        # Walk through arguments and apply required translations
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, trans in algo_plan.required_translations.items():
            bound.arguments[name] = self._add_translation_plan(
                trans, bound.arguments[name]
            )
        args, kwargs = bound.args, bound.kwargs
        # Add resolver if needed by the algorithm
        if algo_plan.algo._include_resolver:
            kwargs["resolver"] = self._resolver
        # Determine return type and add task
        ret = sig.return_annotation
        if getattr(ret, "__origin__", None) == tuple:
            # Use dask.delayed to compute the tuple
            tpl_call = delayed(algo_plan.algo, nout=len(ret.__args__))
            key = (
                f"call-{tokenize(tpl_call, algo_plan, args, kwargs)}",
                f"{algo_plan.algo.abstract_name}",
            )
            tpl = tpl_call(*args, **kwargs, dask_key_name=key)
            # Add extraction tasks for each component
            ret_vals = []
            for i, ret_item in enumerate(ret.__args__):
                extract_func = lambda x, i=i: x[i]
                if type(ret_item) is not type and isinstance(ret_item, ConcreteType):
                    ct = type(ret_item)
                    ph = self._get_placeholder(ct)
                    key = f"[{i}]-{tokenize(ph, tpl, i)}"
                    ret_val = ph.build(key, extract_func, (tpl,))
                else:
                    key = f"[{i}]-{tokenize(tpl, i)}"
                    ret_val = delayed(extract_func)(tpl, dask_key_name=key)
                ret_vals.append(ret_val)
            return tuple(ret_vals)
        elif type(ret) is not type and isinstance(ret, ConcreteType):
            ct = type(ret)
            ph = self._get_placeholder(ct)
            key = (
                f"call-{tokenize(ph, algo_plan, args, kwargs)}",
                f"{algo_plan.algo.abstract_name}",
            )
            return ph.build(key, algo_plan.algo, args, kwargs)
        else:
            # Use dask.delayed instead of a Placeholder
            delayed_call = delayed(algo_plan.algo)
            key = (
                f"call-{tokenize(delayed_call, algo_plan, args, kwargs)}",
                f"{algo_plan.algo.abstract_name}",
            )
            return delayed_call(*args, **kwargs, dask_key_name=key)

    def register(self, *args, **kwargs):
        raise NotImplementedError(
            "Register with the resolver prior to creating a DaskResolver"
        )

    def assert_equal(self, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0):
        if is_dask_collection(obj1):
            obj1 = obj1.compute()
        if is_dask_collection(obj2):
            obj2 = obj2.compute()
        self._resolver.assert_equal(obj1, obj2, rel_tol=rel_tol, abs_tol=abs_tol)

    def translate(self, value, dst_type, **props):
        trans_plan = self.plan.translate(value, dst_type, **props)
        return trans_plan(value, **props)

    def call_algorithm(self, algo_name: str, *args, **kwargs):
        valid_algos = self.find_algorithm_solutions(algo_name, *args, **kwargs)
        if not valid_algos:
            raise TypeError(
                f'No concrete algorithm for "{algo_name}" can be satisfied for the given inputs'
            )
        else:
            # choose the solutions requiring the fewest translations
            plan = valid_algos[0]
            return plan(*args, **kwargs)
