from typing import Union, AnyStr, Tuple, Generator, Iterable
import math
from metagraph.core.resolver import Resolver, Dispatcher
from metagraph import ConcreteType
from metagraph.core.plugin import ConcreteAlgorithm


class UnsatisfiableAlgorithmError(Exception):
    pass


def apply_all(
    resolver: Resolver, algo: Union[Dispatcher, AnyStr], *args, **kwargs
) -> Generator[Tuple[ConcreteAlgorithm, ConcreteType], None, None]:
    """
    Returns a generator which loops over all registered concrete algorithms
    The generator returns a tuple of (ConcreteAlgorithm, returned ConcreteType)

    Typical usage is to first call `apply_all` and pass the result to `verify_all`

    :param resolver: Resolver
    :param algo: abstract algorithm (resolver.algo.path.to.algo or 'path.to.algo')
    :param args: positional parameters passed to algo
    :param kwargs: keyword parameters passed to algo
    :return: Generator of tuples (ConcreteAlgorithm, returned ConcreteType)
    """
    if type(algo) is Dispatcher:
        algo = algo._algo_name
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
    # Run each algorithm and return result in a generator
    for plan in plans:
        ret_val = plan(*args, **kwargs)
        yield plan.algo, ret_val


def verify_all(
    resolver: Resolver,
    expected_val: ConcreteType,
    algo_results: Iterable[Tuple[ConcreteAlgorithm, ConcreteType]],
):
    """
    Verifies that all ret_vals match expected_val, once translated to the correct type

    Typical usage is to first call `apply_all` and pass the result to `verify_all`

    :param resolver: Resolver
    :param expected_val: ConcreteType
    :param algo_results: Generator of tuples (ConcreteAlgorithm, returned ConcreteType)
    :return:
    """
    expected_type = resolver.class_to_concrete.get(
        type(expected_val), type(expected_val)
    )
    for algo, ret_val in algo_results:
        algo_path = f"{algo.func.__module__}.{algo.func.__qualname__}"
        if issubclass(expected_type, ConcreteType):
            try:
                compare_val = resolver.translate(ret_val, type(expected_val))
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
