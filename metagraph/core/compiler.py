import logging
from dataclasses import dataclass
from typing import List, Dict, Hashable, Optional

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

    # which nodes are compilable concrete_algorithms?
    compilable_keys = set()
    for key in dsk.keys():
        task_callable = dsk[key][0]

        if (
            isinstance(task_callable, DelayedAlgo)
            and task_callable.algo._compiler == compiler
        ):
            compilable_keys.add(key)

    non_compilable_keys = set(dsk.keys()) - compilable_keys

    subgraphs = []
    # keep trying to build chains until no compilable tasks are left
    keys_left = compilable_keys.copy()
    while len(keys_left) > 0:
        key = keys_left.pop()
        chain = [key]
        chain_extended = True
        while chain_extended:
            next_compilable_tasks = dependents[chain[-1]] & compilable_keys
            prior_compilable_tasks = dependencies[chain[0]] & compilable_keys

            # Can we extend the chain?
            chain_extended = False

            if len(next_compilable_tasks) == 1:
                candidate = next_compilable_tasks.pop()
                # ensure this task does not have multiple compileable dependencies
                if len(dependencies[candidate] & compilable_keys) == 1:
                    chain.append(candidate)
                    keys_left.remove(chain[-1])
                    chain_extended = True

            if len(prior_compilable_tasks) == 1:
                chain.insert(0, prior_compilable_tasks.pop())
                keys_left.remove(chain[0])
                chain_extended = True

        if len(chain) >= chain_threshold:
            # collect all the inputs to the tasks in this chain
            inputs = set()
            chain_keys = set(chain)
            for key in chain:
                inputs |= dependencies[key] - chain_keys
            tasks = {key: dsk[key] for key in chain}
            subgraphs.append(
                DaskSubgraph(tasks=tasks, input_keys=list(inputs), output_key=chain[-1])
            )

    return subgraphs


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
