import metagraph.core.compiler as mg_compiler
from dask.delayed import delayed


def test_dask_subgraph():
    @delayed
    def func1(x):
        return x + 1

    z = func1(func1(1))

    subgraph = mg_compiler.DaskSubgraph(tasks=z.dask, input_keys=[], output_key=z.key)
    assert len(subgraph.tasks) == 2
    assert len(subgraph.input_keys) == 0
    assert isinstance(subgraph.output_key, str)
