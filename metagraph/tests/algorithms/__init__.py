from typing import Union, AnyStr
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

    def assert_equals(self, expected_val: ConcreteType):
        """
        Verifies that each concrete algorithm's output matches expected_val, once translated to the correct type

        :param expected_val: ConcreteType
        """
        expected_type = self.resolver.class_to_concrete.get(
            type(expected_val), type(expected_val)
        )
        for plan in self.plans:
            algo_path = f"{plan.algo.func.__module__}.{plan.algo.func.__qualname__}"
            ret_val = plan(*self._args, **self._kwargs)
            if issubclass(expected_type, ConcreteType):
                try:
                    compare_val = self.resolver.translate(ret_val, type(expected_val))
                    assert expected_type.compare_objects(
                        compare_val, expected_val
                    ), f"{algo_path} failed comparison check"
                except TypeError:
                    raise UnsatisfiableAlgorithmError(
                        f"[{algo_path}] Unable to convert returned type {type(ret_val)} "
                        f"into type {type(expected_val)} for comparison"
                    )
            else:
                # Normal Python type
                if expected_type is float:
                    assert math.isclose(
                        ret_val, expected_val
                    ), f"[{algo_path}] {ret_val} is not close to {expected_val}"
                else:
                    assert (
                        ret_val == expected_val
                    ), f"[{algo_path}] {ret_val} != {expected_val}"
