from typing import Union, AnyStr, Callable, Tuple
import math
from metagraph import ConcreteType
from metagraph.core.resolver import Resolver, Dispatcher
from dask import is_dask_collection


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
        return results

    def normalize(self, desired_type):
        """
        Convert results into desired type (handling tuple types if required)
        Returns a new MultiResult object
        """
        if self._normalized:
            return self

        if type(desired_type) is tuple and self._length != len(desired_type):
            raise TypeError(
                f"Cannot normalize results of length {self._length} into something of length {len(desired_type)}"
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

    def assert_equals(
        self,
        expected_val: Union[ConcreteType, Tuple[ConcreteType]],
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        return self._verifier.assert_equals(
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

        all_concrete_algos = set(self.resolver.concrete_algorithms[algo])
        plans = self.resolver.find_algorithm_solutions(algo, *args, **kwargs)
        # Check if any concrete algorithm failed to find a valid plan
        for plan in plans:
            all_concrete_algos.remove(plan.algo)
        if all_concrete_algos:
            missing_algos = [
                f"{algo.func.__module__}.{algo.func.__qualname__}"
                for algo in all_concrete_algos
            ]
            if len(missing_algos) == 1:
                missing_algos = missing_algos[0]
            else:
                missing_algos = f"[{', '.join(missing_algos)}]"

            raise UnsatisfiableAlgorithmError(f"No plan found for {missing_algos}")

        results = {}
        for plan in plans:
            algo_path = f"{plan.algo.func.__module__}.{plan.algo.func.__qualname__}"
            try:
                ret_val = plan(*args, **kwargs)

                # Compute any lazy objects
                if is_dask_collection(ret_val):
                    ret_val = ret_val.compute()
                elif type(ret_val) is tuple:
                    ret_val = tuple(
                        x.compute() if is_dask_collection(x) else x for x in ret_val
                    )
                results[algo_path] = ret_val
            except Exception:
                print(f"Failed for {algo_path}")
                raise

        return MultiResult(self, results)

    def _continue_computation(self, algo: Union[Dispatcher, AnyStr], args, kwargs):
        raise NotImplementedError()

    def _translate_atomic_type(self, value, dst_type, algo_path):
        try:
            if (
                issubclass(dst_type, ConcreteType)
                or dst_type in self.resolver.class_to_concrete
            ):
                translated_value = self.resolver.translate(value, dst_type)
                if is_dask_collection(translated_value):
                    translated_value = translated_value.compute()
            else:
                translated_value = value
        except TypeError:
            raise UnsatisfiableAlgorithmError(
                f"[{algo_path}] Unable to convert returned type {type(value)} "
                f"into type {dst_type} for comparison"
            )
        return translated_value

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
                    ret_val = (result,)
                for item in ret_val:
                    if hasattr(item, "value"):
                        print(item.value)
                    else:
                        print(item)
                raise

    def assert_equals(
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
        expected_type = self.resolver.class_to_concrete.get(
            type(expected_val), type(expected_val)
        )
        if issubclass(expected_type, ConcreteType):
            try:
                if not expected_type.is_typeclass_of(val):
                    raise TypeError(f"`val` must be {expected_type}, not {type(val)}")
                self.resolver.assert_equal(
                    val, expected_val, rel_tol=rel_tol, abs_tol=abs_tol
                )
            except AssertionError:
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
