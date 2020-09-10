from typing import Union, AnyStr, Callable, Tuple
import math
from metagraph import ConcreteType
from metagraph.core.resolver import Resolver, Dispatcher


class UnsatisfiableAlgorithmError(Exception):
    pass


class MultiVerify:
    def __init__(
        self, resolver: Resolver, algo: Union[Dispatcher, AnyStr], *args, **kwargs
    ):
        """
        Object which calls `algo` for each concrete type, translating inputs as needed.
        No work is actually performed until `.assert_equals(val)` is called.

        :param resolver: Resolver
        :param algo: abstract algorithm (resolver.algo.path.to.algo or 'path.to.algo')
        :param args: positional parameters passed to algo
        :param kwargs: keyword parameters passed to algo
        """
        if type(algo) is Dispatcher:
            algo = algo._algo_name

        self.resolver = resolver
        self.algo = algo
        self._args = args
        self._kwargs = kwargs

        all_concrete_algos = set(resolver.concrete_algorithms[algo])
        plans = resolver.find_algorithm_solutions(algo, *args, **kwargs)
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

        self.plans = plans

    def _translate_atomic_type(self, value, dst_type, algo_path):
        try:
            if (
                issubclass(dst_type, ConcreteType)
                or dst_type in self.resolver.class_to_concrete
            ):
                translated_value = self.resolver.translate(value, dst_type)
            else:
                translated_value = value
        except TypeError:
            raise UnsatisfiableAlgorithmError(
                f"[{algo_path}] Unable to convert returned type {type(value)} "
                f"into type {dst_type} for comparison"
            )
        return translated_value

    def custom_compare(self, cmp_func: Callable, expected_type=None):
        """
        Calls cmp_func sequentially, passing in each concrete algorithm's output.
        This allows a customized way to verify the result of algorithms.
        If expected_type is provided, each algorithm's output will be translated to this type prior to
            passing it into cmp_func.
        If the algorithm has multiple outputs, cmp_func will be given all outputs at once. No attempt
            is made to loop through tuples.
        """
        for plan in self.plans:
            algo_path = f"{plan.algo.func.__module__}.{plan.algo.func.__qualname__}"
            try:
                ret_val = plan(*self._args, **self._kwargs)

                # Convert ret_val into correct type (handling tuple return types if required)
                if type(expected_type) is tuple:
                    assert len(expected_type) == len(
                        ret_val
                    ), f"[{algo_path}] {ret_val} is not the same length as {expected_type}"
                    rv = []
                    for expected_type_elem, ret_val_elem in zip(expected_type, ret_val):
                        translated_ret_val_elem = self._translate_atomic_type(
                            ret_val_elem, expected_type_elem, algo_path
                        )
                        rv.append(translated_ret_val_elem)
                    ret_val = tuple(rv)
                elif expected_type is not None:
                    ret_val = self._translate_atomic_type(
                        ret_val, expected_type, algo_path
                    )
                try:
                    cmp_func(ret_val)
                except Exception:
                    print("Performing custom compare against:")

                    def _print_ret_val(item):
                        if hasattr(item, "value"):
                            print(item.value)
                        else:
                            print(item)

                    if isinstance(ret_val, tuple):
                        for ret_val_elem in ret_val:
                            _print_ret_val(ret_val_elem)
                    else:
                        _print_ret_val(ret_val)
                    raise
            except Exception:
                print(f"Failed for {algo_path}")
                raise

    def assert_equals(
        self,
        expected_val: Union[ConcreteType, Tuple[ConcreteType]],
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        """
        Verifies that each concrete algorithm's output matches expected_val, once translated to the correct type

        :param expected_val: ConcreteType
        """
        for plan in self.plans:
            algo_path = f"{plan.algo.func.__module__}.{plan.algo.func.__qualname__}"
            try:
                ret_val = plan(*self._args, **self._kwargs)
                if type(expected_val) is tuple:
                    assert len(expected_val) == len(
                        ret_val
                    ), f"[{algo_path}] {ret_val} is not the same length as {expected_val}"
                    for expected_val_elem, ret_val_elem in zip(expected_val, ret_val):
                        self._compare_values(
                            expected_val_elem, ret_val_elem, algo_path, rel_tol, abs_tol
                        )
                else:
                    self._compare_values(
                        expected_val, ret_val, algo_path, rel_tol, abs_tol
                    )
            except Exception:
                print(f"Failed for {algo_path}")
                raise

    def _compare_values(self, expected_val, ret_val, algo_path, rel_tol, abs_tol):
        expected_type = self.resolver.class_to_concrete.get(
            type(expected_val), type(expected_val)
        )
        if issubclass(expected_type, ConcreteType):
            compare_val = self._translate_atomic_type(
                ret_val, type(expected_val), algo_path
            )
            try:
                if not expected_type.is_typeclass_of(compare_val):
                    raise TypeError(
                        f"compare value must be {expected_type}, not {type(compare_val)}"
                    )
                self.resolver.assert_equal(
                    compare_val, expected_val, rel_tol=rel_tol, abs_tol=abs_tol
                )
            except AssertionError:
                print(f"compare_val        {compare_val}")
                print(f"compare_val.value  {compare_val.value}")
                print(f"expected_val       {expected_val}")
                print(f"expected_val.value {expected_val.value}")
                # breakpoint()
                raise
        else:
            # Normal Python type
            if expected_type is float:
                assert math.isclose(
                    ret_val, expected_val, rel_tol=rel_tol, abs_tol=abs_tol
                ), f"[{algo_path}] {ret_val} is not close to {expected_val}"
            else:
                assert (
                    ret_val == expected_val
                ), f"[{algo_path}] {ret_val} != {expected_val}"
