import logging
from dataclasses import dataclass
from typing import List, Dict, Hashable, Optional, Tuple, Generator
from functools import reduce

from dask.core import get_deps

from metagraph.core.plugin import ConcreteAlgorithm, Compiler, CompileError
from metagraph.core.dask.placeholder import DelayedAlgo


@dataclass
class DaskSubgraph:
    """A subgraph of a larger Dask task graph.

    Currently subgraph must have 0 or more inputs and only 1 output
    """

    tasks: dict
    input_keys: List[Hashable]
    output_key: Hashable


def extract_compilable_subgraphs(
    dsk: Dict, compiler: str, include_singletons=True
) -> List[DaskSubgraph]:
    """Find compilable subgraphs in this Dask task graph.

    Currently only works with one compiler at a time, and only will return
    linear chains of compilable tasks.  If include_singletons is True,
    returned chains may be of length 1.  If False, the chain length must be >1.
    """

    if include_singletons:
        chain_threshold = 1
    else:
        chain_threshold = 2
    dependencies, dependents = get_deps(dsk)

    compilable_keys, non_compilable_keys = _get_compilable_dask_keys(dsk, compiler)

    if len(compilable_keys) == 0:
        return []

    subgraphs = []
    ordered_keys = _topologically_sorted_dask_keys(
        compilable_keys, dependencies, dependents
    )
    key = next(ordered_keys)
    current_chain = [key]

    def _note_subgraph(chain):
        output_key = chain[-1]
        chain = set(chain)
        inputs = reduce(
            set.union, (dependencies[chain_key] - chain for chain_key in chain)
        )
        tasks = {chain_key: dsk[chain_key] for chain_key in chain}
        subgraphs.append(
            DaskSubgraph(tasks=tasks, input_keys=list(inputs), output_key=output_key)
        )

    for next_key in ordered_keys:
        next_key_dependencies = dependencies[next_key]
        key_dependents = dependents[key]

        if (
            len(next_key_dependencies) == 1
            and len(key_dependents) == 1
            and next_key in key_dependents
            and key in next_key_dependencies
        ):
            current_chain.append(next_key)
        elif len(current_chain) >= chain_threshold:
            _note_subgraph(current_chain)
            current_chain = [next_key]
        else:
            current_chain = [next_key]
        key = next_key

    if len(current_chain) >= chain_threshold:
        _note_subgraph(current_chain)

    return subgraphs


def _get_compilable_dask_keys(dsk: Dict, compiler: str) -> Tuple[set, set]:

    compilable_keys = set()
    for key in dsk.keys():
        task_callable = dsk[key][0]
        if isinstance(task_callable, DelayedAlgo):
            if task_callable.algo._compiler == compiler:
                compilable_keys.add(key)

    non_compilable_keys = set(dsk.keys()) - compilable_keys

    return compilable_keys, non_compilable_keys


def _topologically_sorted_dask_keys(
    compilable_keys: set,
    dependencies: Dict[Hashable, set],
    dependents: Dict[Hashable, set],
) -> Generator[Hashable, None, None]:
    """
    This is a "greedy" topological sort, e.g. this graph

               a
              / \
             b   d
             |   |
             c   e
        
    might return [a,d,e,b,c] but will never return [a,b,d,c,e] even though the latter is in correct topological order.
    
    This greediness is necessary for the correctness of extract_compilable_subgraphs.
    """

    visited = set()

    def _traverse(key):
        if key not in visited:
            visited.add(key)
            parent_keys = dependencies[key]
            for parent_key in parent_keys:
                yield from _traverse(parent_key)
            yield key

    output_keys = filter(lambda key: len(dependents[key]) == 0, compilable_keys)
    for output_key in output_keys:
        yield from _traverse(output_key)

    return


def compile_subgraphs(dsk, keys, compiler: Compiler):
    """Return a modified dask graph with compilable subgraphs fused together."""

    subgraphs = extract_compilable_subgraphs(dsk, compiler=compiler.name)
    if len(subgraphs) == 0:
        return dsk  # no change, nothing to compile

    # make a new graph we can mutate
    new_dsk = dsk.copy()
    for subgraph in subgraphs:
        try:
            fused_func = compiler.compile_subgraph(
                subgraph.tasks, subgraph.input_keys, subgraph.output_key
            )

            # remove keys for existing tasks in subgraph, including the output task
            for key in subgraph.tasks:
                del new_dsk[key]

            # put the fused task in with the output task's old key
            new_dsk[subgraph.output_key] = (fused_func, *subgraph.input_keys)
        except CompileError as e:
            logging.debug(
                "Unable to compile subgraph with output key: %s",
                subgraph.output_key,
                exc_info=e,
            )
            # continue with graph unchanged to next subgraph

    return new_dsk


def optimize(dsk, keys, *, compiler: Optional[Compiler] = None, **kwargs):
    """Top level optimizer function for Metagraph DAGs"""
    # FUTURE: swap nodes in graph with compilable implementations if they exist?

    if compiler is not None:
        optimized_dsk = compile_subgraphs(dsk, keys, compiler=compiler)

    return optimized_dsk
