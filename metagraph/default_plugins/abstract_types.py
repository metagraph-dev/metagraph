from .. import (
    PluginRegistry,
    AbstractType,
    ConcreteType,
    translator,
    abstract_algorithm,
    concrete_algorithm,
)

reg = PluginRegistry("metagraph_core")


@reg.register
class VectorType(AbstractType):
    name = "Vector"


@reg.register
class SparseVectorType(AbstractType):
    name = "SparseVector"


@reg.register
class MatrixType(AbstractType):
    name = "Matrix"


@reg.register
class SparseMatrixType(AbstractType):
    name = "SparseMatrix"


@reg.register
class DataFrameType(AbstractType):
    name = "DataFrame"


@reg.register
class GraphType(AbstractType):
    name = "Graph"


@reg.register
class WeightedGraphType(GraphType):
    name = "WeightedGraph"
