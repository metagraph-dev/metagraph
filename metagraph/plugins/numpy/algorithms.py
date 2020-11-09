import numpy as np
from functools import reduce
from metagraph import concrete_algorithm, NodeID
from .types import (
    NumpyVectorType,
    NumpyMatrixType,
    NumpyNodeMap,
    NumpyNodeSet,
)
from typing import Any, Callable, Optional
from .. import has_numba

if has_numba:
    import numba


@concrete_algorithm("util.nodeset.choose_random")
def np_nodeset_choose_random(x: NumpyNodeSet, k: int) -> NumpyNodeSet:
    random_elements = np.random.choice(x.value, k, False)
    return NumpyNodeSet(random_elements)


@concrete_algorithm("util.nodeset.from_vector")
def np_nodeset_from_vector(x: NumpyVectorType) -> NumpyNodeSet:
    return NumpyNodeSet(x)


@concrete_algorithm("util.nodemap.sort")
def np_nodemap_sort(
    x: NumpyNodeMap, ascending: bool, limit: Optional[int]
) -> NumpyVectorType:
    positions_of_sorted_values = np.argsort(x.value)
    nodeids_of_sorted_values = x.nodes[positions_of_sorted_values]

    if not ascending:
        nodeids_of_sorted_values = np.flip(nodeids_of_sorted_values)
    if limit:
        nodeids_of_sorted_values = nodeids_of_sorted_values[:limit]
    return nodeids_of_sorted_values


@concrete_algorithm("util.nodemap.select")
def np_nodemap_select(x: NumpyNodeMap, nodes: NumpyNodeSet) -> NumpyNodeMap:
    common_nodes = np.intersect1d(x.nodes, nodes.value)
    index = np.searchsorted(x.nodes, common_nodes)
    selected_data = x.value[index].copy()
    return NumpyNodeMap(selected_data, nodes=common_nodes)


@concrete_algorithm("util.nodemap.filter")
def np_nodemap_filter(x: NumpyNodeMap, func: Callable[[Any], bool]) -> NumpyNodeSet:
    # TODO consider caching this somewhere or enforcing that only vectorized functions are given
    func_vectorized = numba.vectorize(func) if has_numba else np.vectorize(func)
    return NumpyNodeSet(x.nodes[func_vectorized(x.value)].copy())


@concrete_algorithm("util.nodemap.apply")
def np_nodemap_apply(x: NumpyNodeMap, func: Callable[[Any], Any]) -> NumpyNodeMap:
    # TODO consider caching this somewhere or enforcing that only vectorized functions are given
    func_vectorized = numba.vectorize(func) if has_numba else np.vectorize(func)
    return NumpyNodeMap(func_vectorized(x.value), nodes=x.nodes.copy())


@concrete_algorithm("util.nodemap.reduce")
def np_nodemap_reduce(x: NumpyNodeMap, func: Callable[[Any, Any], Any]) -> Any:
    if not isinstance(func, np.ufunc):
        func = np.frompyfunc(func, 2, 1)
    return func.reduce(x.value)


@concrete_algorithm("util.node_embedding.apply")
def np_embedding_apply(
    matrix: NumpyMatrixType, node2row: NumpyNodeMap, nodes: NumpyVectorType
) -> NumpyMatrixType:
    indices = node2row[nodes]
    return matrix[indices]
