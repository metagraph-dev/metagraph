from metagraph import (
    AbstractType,
    ConcreteType,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)


class HyperGraph(AbstractType):
    pass


class CPUHyperGraph(ConcreteType, abstract=HyperGraph):
    pass


class GPUHyperGraph(ConcreteType, abstract=HyperGraph):
    pass


@translator
def cpu_to_gpu_hypergraph(
    src: CPUHyperGraph, **props
) -> GPUHyperGraph:  # pragma: no cover
    pass


@translator
def gpu_to_cpu_hypergraph(
    src: GPUHyperGraph, **props
) -> CPUHyperGraph:  # pragma: no cover
    pass


@abstract_algorithm("hyperstuff.supercluster")
def supercluster(hg: HyperGraph) -> HyperGraph:  # pragma: no cover
    """Make a supercluster from a hypergraph"""
    pass


@concrete_algorithm("hyperstuff.supercluster")
def cpu_supercluster(hg: CPUHyperGraph) -> CPUHyperGraph:  # pragma: no cover
    pass


@concrete_algorithm("hyperstuff.supercluster")
def gpu_supercluster(hg: GPUHyperGraph) -> GPUHyperGraph:  # pragma: no cover
    pass


def abstract_types():
    return [HyperGraph]


def concrete_types():
    return [CPUHyperGraph, GPUHyperGraph]


def translators():
    return [cpu_to_gpu_hypergraph, gpu_to_cpu_hypergraph]


def abstract_algorithms():
    return [supercluster]


def concrete_algorithms():
    return [cpu_supercluster, gpu_supercluster]
