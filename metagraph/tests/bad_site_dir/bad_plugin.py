from metagraph import (
    AbstractType,
    ConcreteType,
    Wrapper,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)


class HyperGraphType(AbstractType):
    pass


class CPUHyperGraphType(ConcreteType, abstract=HyperGraphType):
    pass


class GPUHyperGraph(Wrapper, abstract=HyperGraphType):
    class TypeMixin:
        pass


@translator
def cpu_to_gpu_hypergraph(
    src: CPUHyperGraphType, **props
) -> GPUHyperGraph:  # pragma: no cover
    pass


@translator
def gpu_to_cpu_hypergraph(
    src: GPUHyperGraph, **props
) -> CPUHyperGraphType:  # pragma: no cover
    pass


@abstract_algorithm("hyperstuff.supercluster")
def supercluster(hg: HyperGraphType) -> HyperGraphType:  # pragma: no cover
    """Make a supercluster from a hypergraph"""
    pass


@concrete_algorithm("hyperstuff.supercluster")
def cpu_supercluster(hg: CPUHyperGraphType) -> CPUHyperGraphType:  # pragma: no cover
    pass


@concrete_algorithm("hyperstuff.supercluster")
def gpu_supercluster(hg: GPUHyperGraph) -> GPUHyperGraph:  # pragma: no cover
    pass


def find_plugins():
    return {
        "plugin1": {
            "abstract_types": {HyperGraphType},
            "concrete_types": {CPUHyperGraphType},
            "wrappers": {GPUHyperGraph},
            "translators": {cpu_to_gpu_hypergraph, gpu_to_cpu_hypergraph},
            "abstract_algorithms": {supercluster},
            "concrete_algorithms": {cpu_supercluster, gpu_supercluster},
        }
    }
