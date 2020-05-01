from typing import Union, AnyStr, Callable
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
                    for expected_type_elem, ret_val_elem in zip(expected_type, ret_val):
                        rv = []
                        try:
                            rv.append(
                                self.resolver.translate(
                                    ret_val_elem, expected_type_elem
                                )
                            )
                        except TypeError:
                            raise UnsatisfiableAlgorithmError(
                                f"[{algo_path}] Unable to convert returned type {type(ret_val_elem)} "
                                f"into type {expected_type_elem} for comparison"
                            )
                        ret_val = tuple(rv)
                elif expected_type is not None:
                    try:
                        ret_val = self.resolver.translate(ret_val, expected_type)
                    except TypeError:
                        raise UnsatisfiableAlgorithmError(
                            f"[{algo_path}] Unable to convert returned type {type(ret_val)} "
                            f"into type {expected_type} for comparison"
                        )

                try:
                    cmp_func(ret_val)
                except Exception:
                    print("Performing custom compare against:")
                    print(ret_val)
                    print(ret_val.value)
                    raise
            except Exception:
                print(f"Failed for {algo_path}")
                raise

    def assert_equals(self, expected_val: ConcreteType, rel_tol=1e-9, abs_tol=0.0):
        """
        Verifies that each concrete algorithm's output matches expected_val, once translated to the correct type

        :param expected_val: ConcreteType
        """
        for plan in self.plans:
            algo_path = f"{plan.algo.func.__module__}.{plan.algo.func.__qualname__}"
            try:
                ret_val = plan(*self._args, **self._kwargs)
                if type(expected_val) != tuple:
                    self._compare_values(
                        expected_val, ret_val, algo_path, rel_tol, abs_tol
                    )
                else:
                    assert len(expected_val) == len(
                        ret_val
                    ), f"[{algo_path}] {ret_val} is not the same length as {expected_val}"
                    for expected_val_elem, ret_val_elem in zip(expected_val, ret_val):
                        self._compare_values(
                            expected_val_elem, ret_val_elem, algo_path, rel_tol, abs_tol
                        )
            except Exception:
                print(f"Failed for {algo_path}")
                raise

    def _compare_values(self, expected_val, ret_val, algo_path, rel_tol, abs_tol):
        expected_type = self.resolver.class_to_concrete.get(
            type(expected_val), type(expected_val)
        )
        if issubclass(expected_type, ConcreteType):
            try:
                compare_val = self.resolver.translate(ret_val, type(expected_val))
                if not expected_type.compare_objects(
                    compare_val, expected_val, rel_tol=rel_tol, abs_tol=abs_tol
                ):
                    print(compare_val)
                    print(compare_val.value)
                    print(expected_val)
                    print(expected_val.value)
                    raise AssertionError(f"{algo_path} failed comparison check")
            except TypeError:
                raise UnsatisfiableAlgorithmError(
                    f"[{algo_path}] Unable to convert returned type {type(ret_val)} "
                    f"into type {type(expected_val)} for comparison"
                )
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
