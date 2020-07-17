import random
import operator
from functools import reduce
import numpy as np
from ..numpy.types import NumpyVector
from metagraph import concrete_algorithm, NodeID
from .types import PythonNodeSet, PythonNodeMap
from typing import Tuple, Iterable, Any, Callable, Optional


@concrete_algorithm("util.nodeset.choose_random")
def python_nodeset_choose_random(x: PythonNodeSet, k: int) -> PythonNodeSet:
    return PythonNodeSet(set(random.sample(x.value, k)))


@concrete_algorithm("util.nodemap.sort")
def python_nodemap_sort(
    x: PythonNodeMap, ascending: bool, limit: Optional[int]
) -> NumpyVector:
    sorted_items = sorted(
        x.value.items(), key=operator.itemgetter(1), reverse=(not ascending)
    )
    if limit:
        sorted_items = sorted_items[:limit]
    sorted_keys = np.array(list(map(operator.itemgetter(0), sorted_items)))
    return NumpyVector(sorted_keys)


@concrete_algorithm("util.nodemap.select")
def python_nodemap_select(x: PythonNodeMap, nodes: PythonNodeSet) -> PythonNodeMap:
    return PythonNodeMap({node_id: x.value[node_id] for node_id in nodes.value})


@concrete_algorithm("util.nodemap.filter")
def python_nodemap_filter(
    x: PythonNodeMap, func: Callable[[Any], bool]
) -> PythonNodeSet:
    return PythonNodeSet({key for key, value in x.value.items() if func(value)})


@concrete_algorithm("util.nodemap.apply")
def python_nodemap_apply(x: PythonNodeMap, func: Callable[[Any], Any]) -> PythonNodeMap:
    return PythonNodeMap({key: func(value) for key, value in x.value.items()})


@concrete_algorithm("util.nodemap.reduce")
def python_nodemap_reduce(x: PythonNodeMap, func: Callable[[Any, Any], Any]) -> Any:
    return reduce(func, x.value.values())
