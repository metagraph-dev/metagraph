from .. import (
    PluginRegistry,
    AbstractType,
    ConcreteType,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)
from . import registry as reg


@reg.register
class DenseVector(AbstractType):
    pass


@reg.register
class SparseVector(AbstractType):
    pass


@reg.register
class DenseMatrix(AbstractType):
    pass


@reg.register
class SparseMatrix(AbstractType):
    pass


@reg.register
class DataFrame(AbstractType):
    pass


@reg.register
class Graph(AbstractType):
    pass


@reg.register
class WeightedGraph(Graph):
    pass
