import numpy as np
from functools import reduce
from metagraph import concrete_algorithm, NodeID
from .types import NumpyVector, NumpyNodeMap, NumpyNodeSet
from typing import Any, Callable, Optional
from .. import has_numba

if has_numba:
    import numba


@concrete_algorithm("util.nodeset.choose_random")
def np_nodeset_choose_random(x: NumpyNodeSet, k: int) -> NumpyNodeSet:
    values = np.flatnonzero(x.mask) if x.mask is not None else x.node_array
    random_elements = np.random.choice(np.array(values), k, False)
    random_elements.sort()
    return NumpyNodeSet(random_elements)


@concrete_algorithm("util.nodeset.from_vector")
def np_nodeset_from_vector(x: NumpyVector) -> NumpyNodeSet:
    data = x.value
    if x.mask is not None:
        data = data[x.mask]
    return NumpyNodeSet(node_ids=data)


@concrete_algorithm("util.nodemap.sort")
def np_nodemap_sort(
    x: NumpyNodeMap, ascending: bool, limit: Optional[int]
) -> NumpyVector:
    if x.id2pos is not None:
        positions_of_sorted_values = np.argsort(x.value)
        ids_of_sorted_values = x.pos2id[positions_of_sorted_values]
    elif x.mask is not None:
        ids_of_sorted_values = np.argsort(x.value)
        missing_value_ids = np.flatnonzero(~x.mask)
        ids_of_sorted_values = np.setdiff1d(
            ids_of_sorted_values, missing_value_ids, assume_unique=True
        )
    else:
        ids_of_sorted_values = np.argsort(x.value)
    if not ascending:
        ids_of_sorted_values = np.flip(ids_of_sorted_values)
    if limit:
        ids_of_sorted_values = ids_of_sorted_values[:limit]
    return NumpyVector(ids_of_sorted_values)


@concrete_algorithm("util.nodemap.select")
def np_nodemap_select(x: NumpyNodeMap, nodes: NumpyNodeSet) -> NumpyNodeMap:
    if x.id2pos is not None:
        if nodes.node_array is None:
            select_nodes = np.flatnonzero(nodes.mask)
        else:
            select_nodes = nodes.node_array
        new_pos2id = np.intersect1d(x.pos2id, select_nodes)
        positions_to_keep = np.array([x.id2pos[node_id] for node_id in new_pos2id])
        new_data = x.value[positions_to_keep].copy()
        selected_node_map = NumpyNodeMap(new_data, node_ids=new_pos2id)
    else:
        if nodes.node_array is not None:
            selected_node_map = NumpyNodeMap(
                x.value.copy(),
                mask=x.mask.copy() if x.mask else np.ones(len(x.value), dtype=bool),
            )
            present_value_positions = np.flatnonzero(selected_node_map.mask)
            positions_to_remove = np.setdiff1d(
                present_value_positions, nodes.value, assume_unique=True
            )
            selected_node_map.mask[positions_to_remove] = False
        else:
            if len(nodes_mask) == len(x.mask):
                nodes_mask = nodes.mask
            else:
                nodes_mask = nodes.mask.copy()
                nodes_mask.resize(len(x.mask), refcheck=False)
            new_mask = nodes_mask & x.mask
            selected_node_map = NumpyNodeMap(x.value.copy(), mask=new_mask)
    return selected_node_map


@concrete_algorithm("util.nodemap.filter")
def np_nodemap_filter(x: NumpyNodeMap, func: Callable[[Any], bool]) -> NumpyNodeSet:
    # TODO consider caching this somewhere or enforcing that only vectorized functions are given
    func_vectorized = numba.vectorize(func) if has_numba else np.vectorize(func)
    if x.id2pos is not None:
        filtered_positions = np.flatnonzero(func_vectorized(x.value))
        filtered_ids = x.pos2id[filtered_positions]
    elif x.mask is not None:
        present_values_filter_applied = func_vectorized(x.value[x.mask])
        filter_mask = np.zeros(len(x.value), dtype=bool)
        filter_mask[x.mask] = present_values_filter_applied
        filtered_ids = np.flatnonzero(filter_mask)
    else:
        filtered_ids = np.flatnonzero(func_vectorized(x.value))
    return NumpyNodeSet(filtered_ids)


@concrete_algorithm("util.nodemap.apply")
def np_nodemap_apply(x: NumpyNodeMap, func: Callable[[Any], Any]) -> NumpyNodeMap:
    # TODO consider caching this somewhere or enforcing that only vectorized functions are given
    func_vectorized = numba.vectorize(func) if has_numba else np.vectorize(func)
    if x.id2pos is not None:
        new_node_map = NumpyNodeMap(func_vectorized(x.value), node_ids=x.pos2id.copy())
    elif x.mask is not None:
        results = func_vectorized(new_node_map.value[new_node_map.mask])
        new_data = np.empty_like(x.value, dtype=results.dtype)
        new_data[x.mask] = results
        new_node_map = NumpyNodeMap(new_data, mask=x.mask.copy())
    else:
        new_node_map = NumpyNodeMap(func_vectorized(x.value))
    return new_node_map


@concrete_algorithm("util.nodemap.reduce")
def np_nodemap_reduce(x: NumpyNodeMap, func: Callable[[Any, Any], Any]) -> Any:
    present_values = x.value if x.mask is None else x.value[x.mask]
    if not isinstance(func, np.ufunc):
        func = np.frompyfunc(func, 2, 1)
    return func.reduce(present_values)
