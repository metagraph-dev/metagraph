import random
import operator
from functools import reduce
import numpy as np
from ..numpy.types import NumpyVectorType
from metagraph import concrete_algorithm, NodeID
from .types import PythonNodeSetType, PythonNodeMapType
from typing import Tuple, Iterable, Any, Callable, Optional


@concrete_algorithm("util.nodeset.choose_random")
def python_nodeset_choose_random(x: PythonNodeSetType, k: int) -> PythonNodeSetType:
    return set(random.sample(x, k))


@concrete_algorithm("util.nodemap.sort")
def python_nodemap_sort(
    x: PythonNodeMapType, ascending: bool, limit: Optional[int]
) -> NumpyVectorType:
    sorted_items = sorted(
        x.items(), key=operator.itemgetter(1), reverse=(not ascending)
    )
    if limit:
        sorted_items = sorted_items[:limit]
    sorted_keys = np.array(list(map(operator.itemgetter(0), sorted_items)))
    return sorted_keys


@concrete_algorithm("util.nodemap.select")
def python_nodemap_select(
    x: PythonNodeMapType, nodes: PythonNodeSetType
) -> PythonNodeMapType:
    return {node_id: x[node_id] for node_id in nodes}


@concrete_algorithm("util.nodemap.filter")
def python_nodemap_filter(
    x: PythonNodeMapType, func: Callable[[Any], bool]
) -> PythonNodeSetType:
    return {key for key, value in x.items() if func(value)}


@concrete_algorithm("util.nodemap.apply")
def python_nodemap_apply(
    x: PythonNodeMapType, func: Callable[[Any], Any]
) -> PythonNodeMapType:
    return {key: func(value) for key, value in x.items()}


@concrete_algorithm("util.nodemap.reduce")
def python_nodemap_reduce(x: PythonNodeMapType, func: Callable[[Any, Any], Any]) -> Any:
    return reduce(func, x.values())
