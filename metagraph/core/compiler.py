import logging
from dataclasses import dataclass
from typing import List, Dict, Hashable, Optional, Tuple, Generator
from functools import reduce

from dask.core import get_deps
import dask.optimization

from metagraph.core.plugin import ConcreteAlgorithm, Compiler, CompileError
from metagraph.core.dask.tasks import DelayedAlgo, DelayedJITAlgo


@dataclass
class DaskSubgraph:
    """A subgraph of a larger Dask task graph.

    Currently subgraph must have 0 or more inputs and only 1 output
    """

    tasks: dict
    input_keys: List[Hashable]
    output_key: Hashable


def extract_compilable_subgraphs(
    dsk: Dict, compiler: str, output_keys: List[str], include_singletons=True
) -> List[DaskSubgraph]:
    """Find compilable subgraphs in this Dask task graph.

    Currently only works with one compiler at a time, and only will return
    linear chains of compilable tasks.  If include_singletons is True,
    returned chains may be of length 1.  If False, the chain length must be
    >1.

    If present in a subgraph, tasks corresponding to output_keys can only be
    at the end of a chain.
    """

    if include_singletons:
        chain_threshold = 1
    else:
        chain_threshold = 2
    dependencies, dependents = get_deps(dsk)

    compilable_keys, non_compilable_keys = _get_compilable_dask_keys(dsk, compiler)

    if len(compilable_keys) == 0:
        return []

    output_keys_set = set(output_keys)

    subgraphs = []
    ordered_keys = _dfs_sorted_dask_keys(compilable_keys, dependencies, dependents)
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
            and key not in output_keys_set  # output keys must be at the end of a chain
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


def _dfs_sorted_dask_keys(
    compilable_keys: set,
    dependencies: Dict[Hashable, set],
    dependents: Dict[Hashable, set],
) -> Generator[Hashable, None, None]:

    visited = set()

    def _dfs(key):
        if key not in visited:
            visited.add(key)
            child_keys = dependents[key]
            yield key
            for child_key in child_keys:
                yield from _dfs(child_key)

    input_keys = filter(
        lambda key: len(dependencies[key].intersection(compilable_keys)) == 0,
        compilable_keys,
    )
    for input_key in input_keys:
        yield from _dfs(input_key)

    return


def compile_subgraphs(dsk, output_keys, compiler: Compiler):
    """Return a modified dask graph with compilable subgraphs fused together."""

    subgraphs = extract_compilable_subgraphs(
        dsk, output_keys=output_keys, compiler=compiler.name
    )
    if len(subgraphs) == 0:
        return dsk  # no change, nothing to compile

    # make a new graph we can mutate
    new_dsk = dsk.copy()
    for subgraph in subgraphs:
        try:
            fused_func = compiler.compile_subgraph(
                subgraph.tasks, subgraph.input_keys, subgraph.output_key
            )

            # remember the algorithms being fused and return type
            # this assumes all tasks in the subgraph are DelayedAlgo tasks!
            source_algos = [task[0].algo for task in subgraph.tasks.values()]
            output_task = subgraph.tasks[subgraph.output_key]
            result_type = output_task[0].result_type

            # remove keys for existing tasks in subgraph, including the output task
            for key in subgraph.tasks:
                del new_dsk[key]

            # create a fused task with the output task's old key
            fused_task = DelayedJITAlgo(
                fused_func,
                compiler=compiler.name,
                source_algos=source_algos,
                result_type=result_type,
            )
            new_dsk[subgraph.output_key] = (fused_task, *subgraph.input_keys)
        except CompileError as e:  # pragma: no cover
            logging.debug(
                "Unable to compile subgraph with output key: %s",
                subgraph.output_key,
                exc_info=e,
            )
            # continue with graph unchanged to next subgraph

    return new_dsk


def optimize(dsk, output_keys, **kwargs):
    """Top level optimizer function for Metagraph DAGs"""
    # FUTURE: swap nodes in graph with compilable implementations if they exist?
    optimized_dsk = dsk

    # cull unused nodes
    optimized_dsk, dependencies = dask.optimization.cull(optimized_dsk, output_keys)
    # FUTURE: could speed up extract_compilable_subgraphs by using this dependencies
    # dict to compute dependents as well and passing both dicts into the function
    # so redunant work isn't performed.

    # discover all the compilers referenced in this DAG
    compilers = {}
    for key in optimized_dsk.keys():
        task_callable = optimized_dsk[key][0]
        if isinstance(task_callable, DelayedAlgo):
            if task_callable.algo._compiler is not None:
                compiler_name = task_callable.algo._compiler
                if compiler_name not in compilers:
                    compilers[compiler_name] = task_callable.algo.resolver.compilers[
                        compiler_name
                    ]

    # allow each compiler to optimize the graph
    for compiler in compilers.values():
        optimized_dsk = compile_subgraphs(
            optimized_dsk, output_keys=output_keys, compiler=compiler
        )
    return optimized_dsk
