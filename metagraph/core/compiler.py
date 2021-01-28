from dataclasses import dataclass
from typing import List, Dict, Hashable

from dask.core import get_deps

from metagraph.core.plugin import ConcreteAlgorithm


@dataclass
class DaskSubgraph:
    """A subgraph of a larger Dask task graph.

    Currently subgraph must have 0 or more inputs and only 1 output
    """

    tasks: dict
    input_keys: List[Hashable]
    output_key: Hashable


def extract_compilable_subgraphs(dsk: Dict, compiler: str) -> List[DaskSubgraph]:
    """Find compilable subgraphs in this Dask task graph.

    Currently only works with one compiler at a time, and only will return
    linear chains of compilable tasks with length > 1.  Single compilable
    tasks will be compiled as they are executed.
    """
    dependencies, dependents = get_deps(dsk)

    # which nodes are compilable concrete_algorithms?
    compilable_keys = set()
    for key in dsk.keys():
        task_callable, task_args = dsk[key]
        if (
            isinstanace(task_callable, ConcreteAlgorithm)
            and task_callable._compiler == compiler
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
            next_compilable_tasks = dependents[chain[-1]].intersect(compilable_keys)
            prior_compilable_tasks = dependencies[chain[0]].intersect(compilable_keys)

            # Can we extend the chain?
            chain_extended = False

            if len(next_compilable_tasks) == 1:
                chain.append(next_compilable_tasks.pop())
                keys_left.remove(chain[-1])
                chain_extended = True

            if len(prior_compilable_tasks) == 1:
                chain.insert(0, prior_compilable_tasks.pop())
                keys_left.remove(chain[0])
                chain_extended = True

        if len(chain) > 1:
            # collect all the non-compiled inputs to the tasks in this chain
            inputs = set()
            for key in chain:
                inputs.union(dependencies[key].intersect(non_compilable_keys))
            tasks = {key: dsk[key] for key in chain}
            subgraphs.append(
                DaskSubgraph(tasks=tasks, input_keys=list(inputs), output_key=chain[-1])
            )

    return subgraphs
