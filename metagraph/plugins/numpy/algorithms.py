import numpy as np
from functools import reduce
from metagraph import concrete_algorithm, NodeID
from .types import NumpyVector, NumpyNodeMap, CompactNumpyNodeMap, NumpyNodeSet
from typing import Any, Callable, Optional


@concrete_algorithm("util.nodeset.choose_random")
def np_nodeset_choose_random(x: NumpyNodeSet, k: int) -> NumpyNodeSet:
    return NumpyNodeSet(np.random.choice(np.array(x.value), k, False))


@concrete_algorithm("util.nodemap.sort")
def np_nodemap_sort(
    x: NumpyNodeMap, ascending: bool, limit: Optional[int]
) -> NumpyVector:
    ids_of_sorted_values = np.argsort(x.value)
    if x.missing_mask is not None:
        missing_value_ids = np.argwhere(x.missing_mask).flatten()
        ids_of_sorted_values = np.setdiff1d(
            ids_of_sorted_values, missing_value_ids, assume_unique=True
        )
    if not ascending:
        ids_of_sorted_values = np.flip(ids_of_sorted_values)
    if limit:
        ids_of_sorted_values = ids_of_sorted_values[:limit]
    return NumpyVector(ids_of_sorted_values)


# @concrete_algorithm("util.nodemap.sort")
# def np_compact_nodemap_sort(x: CompactNumpyNodeMap, ascending: bool, limit: Optional[int]) -> NumpyVector:
#     indices_of_sorted_values = np.argsort(x.value)
#     if limit:
#         indices_of_sorted_values = indices_of_sorted_values[:limit]
#     reverse_lookup = {value: key for key, value in x.lookup} # TODO consider caching this
#     ids_of_sorted_values = np.array([reverse_lookup[index] for index in indices_of_sorted_values])
#     if not ascending:
#         ids_of_sorted_values = np.flip(ids_of_sorted_values)
#     return NumpyVector(indices_of_sorted_values)


@concrete_algorithm("util.nodemap.select")
def np_nodemap_select(x: NumpyNodeMap, nodes: NumpyNodeSet) -> NumpyNodeMap:
    selected_node_map = NumpyNodeMap(x.value.copy(), missing_mask=x.missing_mask.copy())
    present_value_indices = np.argwhere(~selected_node_map.missing_mask).flatten()
    indices_to_remove = np.setdiff1d(present_value_indices, nodes.value)
    selected_node_map.missing_mask[indices_to_remove] = True
    return selected_node_map


# @concrete_algorithm("util.nodemap.select")
# def np_compact_nodemap_select(x: CompactNumpyNodeMap, nodes: NumpyNodeSet) -> CompactNumpyNodeMap:
#     new_lookup = {node_id: node_index for node_id, node_index in x.lookup.items() if np.isin(node_id, nodes.value)}
#     retained_indices = np.array(list(new_lookup.values()))
#     new_data = x.value[retained_indices]
#     old_index_to_new_index = retained_indices
#     new_lookup = {node_id: old_index_to_new_index[old_node_index] for node_id, old_node_index in new_lookup.items()}
#     return NumpyNodeMap(new_data, node_lookup=new_lookup)


@concrete_algorithm("util.nodemap.filter")
def np_nodemap_filter(x: NumpyNodeMap, func: Callable[[Any], bool]) -> NumpyNodeSet:
    present_value_indices = np.argwhere(~x.missing_mask).flatten()
    present_values = x.value[present_value_indices]
    func_vectorized = np.vectorize(
        func
    )  # TODO consider caching this somewhere or enforcing that only vectorized functions are given
    filtered_indices = present_value_indices[func_vectorized(present_values)]
    return NumpyNodeSet(filtered_indices)


# @concrete_algorithm("util.nodemap.filter")
# def np_compact_nodemap_filter(x: CompactNumpyNodeMap, func: Callable[[Any], bool]) -> NumpyNodeSet:
#     func_vectorized = np.vectorize(func) # TODO consider caching this somewhere or enforcing that only vectorized functions are given
#     retained_indices = np.argwhere(func_vectorized(x.value)).flatten()
#     new_data = x.value[retained_indices]
#     old_index_to_new_index = retained_indices
#     new_lookup = {node_id: old_index_to_new_index[old_node_index] for node_id, old_node_index in x.lookup.items() if old_node_index in retained_indices}
#     return NumpyNodeMap(new_data, node_lookup=new_lookup)


@concrete_algorithm("util.nodemap.apply")
def np_nodemap_apply(x: NumpyNodeMap, func: Callable[[Any], Any]) -> NumpyNodeMap:
    func_vectorized = np.vectorize(
        func
    )  # TODO consider caching this somewhere or enforcing that only vectorized functions are given
    return NumpyNodeMap(func_vectorized(x.value), missing_mask=x.missing_mask.copy())


# @concrete_algorithm("util.nodemap.apply")
# def np_compact_nodemap_apply(x: CompactNumpyNodeMap, func: Callable[[Any], Any]) -> CompactNumpyNodeMap:
#     func_vectorized = np.vectorize(func) # TODO consider caching this somewhere or enforcing that only vectorized functions are given
#     return CompactNumpyNodeMap(func_vectorized(x.value), node_lookup=x.node_lookup) # TODO copy node_lookup or not?


@concrete_algorithm("util.nodemap.reduce")
def np_nodemap_reduce(x: NumpyNodeMap, func: Callable[[Any, Any], Any]) -> Any:
    present_values = x.value[~x.missing_mask]
    return (
        func.reduce(present_values)
        if isinstance(func, np.ufunc)
        else reduce(func, present_values)
    )


# @concrete_algorithm("util.nodemap.reduce")
# def np_compact_nodemap_reduce(x: NumpyNodeMap, func: Callable[[Any, Any], Any]) -> Any:
#     present_values = x.value
#     return func.reduce(present_values) if isinstance(func, np.ufunc) else reduce(func, present_values)
