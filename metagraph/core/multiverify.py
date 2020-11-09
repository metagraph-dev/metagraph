from typing import Union, AnyStr, Callable, Tuple
import math
import itertools
from metagraph import ConcreteType
from metagraph.core.resolver import Resolver, Dispatcher, ExactDispatcher
from dask import is_dask_collection
import warnings

try:
    import pytest

    has_pytest = True
except ImportError:
    has_pytest = False


def ensure_computed(obj):
    # Compute any lazy objects
    if is_dask_collection(obj):
        obj = obj.compute()
    elif type(obj) is tuple:
        obj = tuple(x.compute() if is_dask_collection(x) else x for x in obj)
    return obj


class UnsatisfiableAlgorithmError(Exception):
    pass


class MultiResult:
    def __init__(self, mv, results, normalized=False):
        self._verifier = mv
        self._results = results
        self._normalized = normalized

        # Inspect results for length consistency
        common_length = None
        for i, result in enumerate(results.values()):
            length = len(result) if type(result) is tuple else None
            if i == 0:
                common_length = length
            else:
                if length != common_length:
                    raise ValueError(
                        f"length mismatch in results: {length} != {common_length}"
                    )
        self._length = common_length

    def __getitem__(self, key):
        if self._length is None:
            raise TypeError("Results are not multi-valued")
        results = {algo: vals[key] for algo, vals in self._results.items()}
        return MultiResult(self._verifier, results, normalized=self._normalized)

    def normalize(self, desired_type):
        """
        Convert results into desired type (handling tuple types if required)
        Returns a new MultiResult object
        """
        if self._normalized:
            return self

        if (
            type(desired_type) is tuple
            and self._length != len(desired_type)
            or (type(desired_type) is not tuple and self._length is not None)
        ):
            desired_len = len(desired_type) if type(desired_type) is tuple else None
            raise TypeError(
                f"Cannot normalize results of length {self._length} into something of length {desired_len}"
            )

        new_results = {}
        for algo_path, ret_val in self._results.items():
            if type(desired_type) is tuple:
                rv = []
                for desired_type_elem, ret_val_elem in zip(desired_type, ret_val):
                    translated_ret_val_elem = self._verifier._translate_atomic_type(
                        ret_val_elem, desired_type_elem, algo_path
                    )
                    rv.append(translated_ret_val_elem)
                ret_val = tuple(rv)
            else:
                ret_val = self._verifier._translate_atomic_type(
                    ret_val, desired_type, algo_path
                )
            new_results[algo_path] = ret_val
        return MultiResult(self._verifier, new_results, normalized=True)

    def custom_compare(self, cmp_func: Callable):
        return self._verifier.custom_compare(self, cmp_func)

    def assert_equal(
        self,
        expected_val: Union[ConcreteType, Tuple[ConcreteType]],
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        return self._verifier.assert_equal(
            self, expected_val, rel_tol=rel_tol, abs_tol=abs_tol
        )


class MultiVerify:
    def __init__(self, resolver: Resolver):
        self.resolver = resolver

    def compute(self, algo: Union[Dispatcher, AnyStr], *args, **kwargs):
        """
        :param algo: abstract algorithm (resolver.algo.path.to.algo or 'path.to.algo')
        :param args: positional parameters passed to algo
        :param kwargs: keyword parameters passed to algo

        Passing in MultiResults to args or kwargs is allowed as long as they have been normalized first.
        Doing so will cause the best algorithm plan for `algo` to be run for every result in the MultiResult,
        effectively making it a continuation of the original `compute` call.

        If multiple MultiResults are present in args and kwargs, all pairs will be run through
        the best algorithm plan for `algo`, multiplying the number of results.
        """
        if type(algo) is Dispatcher:
            algo = algo._algo_name
        if not isinstance(algo, str):
            raise TypeError(
                f'"algo" must be of type `str` or `Dispatcher`, not `{type(algo)}`'
            )

        # Verify no MultiResults in args or kwargs
        abst_algo = self.resolver.abstract_algorithms[algo]
        abst_sig = abst_algo.__signature__
        try:
            bound = abst_sig.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f'Invalid call signature for "{algo}": {e}')
        for name, arg in bound.arguments.items():
            if isinstance(arg, MultiResult):
                raise TypeError(
                    f'Invalid argument "{name}"; may not be a MultiResult. Use `.transform` instead.'
                )

        all_concrete_algos = set(self.resolver.concrete_algorithms[algo])
        if not all_concrete_algos:
            msg = f"No concrete algorithms exist which implement {algo}"
            if has_pytest:
                pytest.skip(msg)
            else:
                warnings.warn(msg)  # pragma: no cover
        plans = self.resolver.find_algorithm_solutions(algo, *args, **kwargs)
        # Check if any concrete algorithm failed to find a valid plan
        for plan in plans:
            all_concrete_algos.remove(plan.algo)
        if all_concrete_algos:
            missing_algos = [
                f"{algo.func.__module__}.{algo.func.__qualname__}"
                for algo in all_concrete_algos
            ]
            missing_algos = (
                missing_algos[0]
                if len(missing_algos) == 1
                else f"[{', '.join(missing_algos)}]"
            )
            raise UnsatisfiableAlgorithmError(f"No plan found for {missing_algos}")

        results = {}
        for plan in plans:
            algo_path = f"{plan.algo.func.__module__}.{plan.algo.func.__qualname__}"
            try:
                ret_val = plan(*args, **kwargs)
                results[algo_path] = ensure_computed(ret_val)
            except Exception:  # pragma: no cover
                print(f"Failed for {algo_path}")
                raise

        return MultiResult(self, results)

    def transform(self, exact_algo: Union[ExactDispatcher, AnyStr], *args, **kwargs):
        """
        :param exact_algo: exact algorithm (resolver.algos.path.to.algo.plugin or 'path.to.algo.plugin')
        :param args: positional parameters passed to algo
        :param kwargs: keyword parameters passed to algo

        At least one MultiResult must exist in args or kwargs. All MultiResults must have been normalized first.

        If multiple MultiResults are present in args and kwargs, all pairs will be run through
        `exact_algo`, multiplying the number of items in the resultant MultiResult.
        """
        if isinstance(exact_algo, str):
            ns = self.resolver.algos
            for name in exact_algo.split("."):
                try:
                    ns = getattr(ns, name)
                except AttributeError:
                    raise TypeError(f'Cannot find exact algorithm "{exact_algo}"')
            exact_algo = ns
        if type(exact_algo) is not ExactDispatcher:
            raise TypeError(
                f'"algo" must be of type `str` or `ExactDispatcher`, not `{type(exact_algo)}`'
            )

        # Verify MultiResults exist somewhere in args or kwargs
        # If multiple MultiResults exist, build an all-pairs combination list
        mr_argnames = []
        sig = exact_algo._algo.__signature__
        bound = sig.bind(*args, **kwargs)
        for name, arg in bound.arguments.items():
            if isinstance(arg, MultiResult):
                if not arg._normalized:
                    raise TypeError(
                        f'"{name}" must be normalized to use in .transform()'
                    )
                if not arg._results:
                    raise ValueError(f'"{name}" has no results')
                mr_argnames.append(name)
        if not mr_argnames:
            raise TypeError(
                f"No MultiResults found in call arguments; .transform() requires at least one MultiResult argument"
            )

        if len(mr_argnames) == 1:
            # Convert algo names into 1-tuples for consistent logic
            name = mr_argnames[0]
            combo_results = {(k,): v for k, v in bound.arguments[name]._results.items()}
        else:
            # Build combinations of MultiResult algo names
            algo_pairs = itertools.product(
                *(bound.arguments[name]._results.keys() for name in mr_argnames)
            )
            combo_results = {}
            for algo_pair in algo_pairs:
                result = []
                for algo, name in zip(algo_pair, mr_argnames):
                    result.append(bound.arguments[name]._results[algo])
                combo_results[algo_pair] = tuple(result)

        # Call exact algorithm for each combination
        output_results = {}
        for algo_pair, results in combo_results.items():
            tmp_bind = sig.bind(*args, **kwargs)
            for algo, name in zip(algo_pair, mr_argnames):
                tmp_bind.arguments[name] = tmp_bind.arguments[name]._results[algo]
            algo_key = algo_pair if len(algo_pair) > 1 else algo_pair[0]
            ret_val = exact_algo(*tmp_bind.args, **tmp_bind.kwargs)
            output_results[algo_key] = ensure_computed(ret_val)

        return MultiResult(self, output_results, normalized=True)

    def _translate_atomic_type(self, value, dst_type, algo_path):
        try:
            if dst_type is not None and (
                issubclass(dst_type, ConcreteType)
                or dst_type in self.resolver.class_to_concrete
            ):
                translated_value = self.resolver.translate(value, dst_type)
            else:
                translated_value = value
        except TypeError:
            raise UnsatisfiableAlgorithmError(
                f"[{algo_path}] Unable to convert type {type(value)} "
                f"into type {dst_type} for comparison"
            )
        return ensure_computed(translated_value)

    def custom_compare(self, multi_result: MultiResult, cmp_func: Callable):
        """
        Calls cmp_func sequentially, passing in each concrete algorithm's output.
        This allows a customized way to verify the result of algorithms.
        Results must be normalized first to ensure a consistent type being passed to cmp_func.
        If the algorithm has multiple outputs, cmp_func will be given all outputs at once. No attempt
            is made to loop through tuples.
        """
        for algo_path, result in multi_result._results.items():
            try:
                cmp_func(result)
            except Exception:
                print(f"[{algo_path}] Performing custom compare against:")
                if not isinstance(result, tuple):
                    result = (result,)
                for item in result:
                    if hasattr(item, "value"):
                        print(item.value)
                    else:
                        print(item)
                raise

    def assert_equal(
        self,
        multi_result: MultiResult,
        expected_val: Union[ConcreteType, Tuple[ConcreteType]],
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        """
        Verifies that each concrete algorithm's output matches expected_val, once translated to the correct type

        :param expected_val: ConcreteType
        """
        # Normalize results
        if type(expected_val) is tuple:
            expected_type = tuple(
                self.resolver.class_to_concrete.get(type(ev), type(ev))
                for ev in expected_val
            )
        else:
            expected_type = self.resolver.class_to_concrete.get(
                type(expected_val), type(expected_val)
            )
        multi_result = multi_result.normalize(expected_type)

        for algo_path, result in multi_result._results.items():
            if type(expected_val) is tuple:
                for expected_val_elem, ret_val_elem in zip(expected_val, result):
                    self.compare_values(
                        expected_val_elem, ret_val_elem, algo_path, rel_tol, abs_tol
                    )
            else:
                self.compare_values(expected_val, result, algo_path, rel_tol, abs_tol)

    def compare_values(self, expected_val, val, algo_path, rel_tol=1e-9, abs_tol=0.0):
        expected_val = ensure_computed(expected_val)
        val = ensure_computed(val)

        try:
            expected_type = self.resolver.typeclass_of(expected_val)
        except TypeError:
            # Assume this is a normal Python type
            expected_type = type(expected_val)

        if issubclass(expected_type, ConcreteType):
            try:
                if not expected_type.is_typeclass_of(val):
                    raise TypeError(f"`val` must be {expected_type}, not {type(val)}")
                self.resolver.assert_equal(
                    val, expected_val, rel_tol=rel_tol, abs_tol=abs_tol
                )
            except AssertionError:
                print(f"assert_equal failed for {algo_path}")
                print(f"val {val}")
                if hasattr(val, "value"):
                    print(f"val.value {val.value}")
                print(f"expected_val {expected_val}")
                if hasattr(expected_val, "value"):
                    print(f"expected_val.value {expected_val.value}")
                # breakpoint()
                raise
        else:
            # Normal Python type
            if expected_type is float:
                assert math.isclose(
                    val, expected_val, rel_tol=rel_tol, abs_tol=abs_tol
                ), f"[{algo_path}] {val} is not close to {expected_val}"
            else:
                assert val == expected_val, f"[{algo_path}] {val} != {expected_val}"
